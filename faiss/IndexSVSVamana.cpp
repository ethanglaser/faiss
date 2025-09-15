/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexSVSVamana.h>

#include <svs/core/data.h>
#include <svs/extensions/vamana/scalar.h>
#include <svs/orchestrators/dynamic_vamana.h>

#include <faiss/MetricType.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

namespace {
svs::index::vamana::VamanaSearchParameters make_search_parameters(
        const IndexSVSVamana* idx,
        const SearchParameters* params) {
    FAISS_THROW_IF_NOT(idx);
    FAISS_THROW_IF_NOT(idx->impl);

    auto search_window_size = idx->search_window_size;
    auto search_buffer_capacity = idx->search_buffer_capacity;

    if (params != nullptr) {
        auto* svs_params =
                dynamic_cast<const SearchParametersSVSVamana*>(params);
        if (svs_params != nullptr) {
            if (svs_params->search_window_size > 0)
                search_window_size = svs_params->search_window_size;
            if (svs_params->search_buffer_capacity > 0)
                search_buffer_capacity = svs_params->search_buffer_capacity;
        }
    }

    return idx->impl->get_search_parameters().buffer_config(
            {search_window_size, search_buffer_capacity});
}
} // namespace

IndexSVSVamana::IndexSVSVamana() : Index{} {}

IndexSVSVamana::IndexSVSVamana(
        idx_t d,
        size_t degree,
        MetricType metric,
        StorageKind storage)
        : Index(d, metric), graph_max_degree{degree}, storage_kind{storage} {
    prune_to = graph_max_degree < 4 ? graph_max_degree : graph_max_degree - 4;
    alpha = metric == METRIC_L2 ? 1.2f : 0.95f;
}

IndexSVSVamana::~IndexSVSVamana() {
    if (impl) {
        delete impl;
        impl = nullptr;
    }
}

void IndexSVSVamana::add(idx_t n, const float* x) {
    if (!impl) {
        init_impl(n, x);
        return;
    }

    // construct sequential labels
    std::vector<size_t> labels(n);

    svs::threads::parallel_for(
            impl->get_threadpool_handle(),
            svs::threads::StaticPartition(n),
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    labels[i] = ntotal + i;
                }
            });
    ntotal += n;

    auto data = svs::data::ConstSimpleDataView<float>(x, n, d);
    impl->add_points(data, labels);
}

void IndexSVSVamana::reset() {
    if (impl) {
        delete impl;
        impl = nullptr;
    }
    ntotal = 0;
    ntotal_soft_deleted = 0;
}

void IndexSVSVamana::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(impl);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    auto sp = make_search_parameters(this, params);

    // Simple search
    if (params == nullptr || params->sel == nullptr) {
        auto queries = svs::data::ConstSimpleDataView<float>(x, n, d);

        // TODO: faiss use int64_t as label whereas SVS uses size_t?
        auto results = svs::QueryResultView<size_t>{
                svs::MatrixView<size_t>{
                        svs::make_dims(n, k),
                        static_cast<size_t*>(static_cast<void*>(labels))},
                svs::MatrixView<float>{svs::make_dims(n, k), distances}};
        impl->search(results, queries, sp);
        return;
    }

    // Selective search with IDSelector
    auto old_sp = impl->get_search_parameters();
    impl->set_search_parameters(sp);

    auto search_closure = [&](const auto& range, uint64_t SVS_UNUSED(tid)) {
        for (auto i : range) {
            // For every query
            auto query = std::span(x + i * d, d);
            auto curr_distances = std::span(distances + i * k, k);
            auto curr_labels = std::span(labels + i * k, k);

            auto iterator = impl->batch_iterator(query);
            idx_t found = 0;
            do {
                iterator.next(k);
                for (auto& neighbor : iterator.results()) {
                    if (params->sel->is_member(neighbor.id())) {
                        curr_distances[found] = neighbor.distance();
                        curr_labels[found] = neighbor.id();
                        found++;
                        if (found == k) {
                            break;
                        }
                    }
                }
            } while (found < k && !iterator.done());
            // Pad with -1s
            for (; found < k; ++found) {
                curr_distances[found] = -1;
                curr_labels[found] = -1;
            }
        }
    };

    // Do not use Thread Pool from index because SVS TP calls are blocking
    // and may be blocked by nested calls
    auto threadpool = svs::threads::OMPThreadPool(
            std::min(n, idx_t(omp_get_max_threads())));

    svs::threads::parallel_for(
            threadpool, svs::threads::StaticPartition{n}, search_closure);

    impl->set_search_parameters(old_sp);
}

void IndexSVSVamana::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(impl);
    FAISS_THROW_IF_NOT(radius > 0);
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(result->nq == n);

    auto sp = make_search_parameters(this, params);
    auto old_sp = impl->get_search_parameters();
    impl->set_search_parameters(sp);

    // Using ResultHandler makes no sense due to it's complexity, overhead and
    // missed features; e.g. add_result() does not indicate whether result added
    // or not - we have to manually manage threshold comparison and id
    // selection.

    // Prepare output buffers
    std::vector<std::vector<svs::Neighbor<size_t>>> all_results(n);
    // Reserve space for allocation to avoid multiple reallocations
    // Use search_buffer_capacity as a heuristic
    const auto result_capacity = sp.buffer_config_.get_total_capacity();
    for (auto& res : all_results) {
        res.reserve(search_buffer_capacity); // Reserve space for elements
    }
    auto sel = params != nullptr ? params->sel : nullptr;

    std::function<bool(float, float)> cmp = std::greater<float>{};
    if (is_similarity_metric(metric_type)) {
        cmp = std::less<float>{};
    }

    // Set iterator batch size to search window size
    auto batch_size = sp.buffer_config_.get_search_window_size();

    auto range_search_closure = [&](const auto& range,
                                    uint64_t SVS_UNUSED(tid)) {
        for (auto i : range) {
            // For every query
            auto query = std::span(x + i * d, d);

            auto iterator = impl->batch_iterator(query);
            bool in_range = true;

            do {
                iterator.next(batch_size);
                for (auto& neighbor : iterator.results()) {
                    // SVS comparator functor returns true if the first distance
                    // is 'closer' than the second one
                    in_range = compare(neighbor.distance(), radius);
                    if (in_range) {
                        // Selective search with IDSelector
                        if (select(neighbor.id())) {
                            all_results[i].push_back(neighbor);
                        }
                    } else {
                        // Since iterator.results() are ordered by distance, we
                        // can stop processing
                        break;
                    }
                }
            } while (in_range && !iterator.done());
        }
    };

    // Do not use TP from index 'cause it may be blocked by nested calls
    auto threadpool = svs::threads::OMPThreadPool(
            std::min(n, idx_t(omp_get_max_threads())));

    svs::threads::parallel_for(
            threadpool, svs::threads::StaticPartition{n}, range_search_closure);

    // RangeSearchResult .ctor() allows unallocated lims
    if (result->lims == nullptr) {
        result->lims = new size_t[result->nq + 1];
    }

    std::transform(
            all_results.begin(),
            all_results.end(),
            result->lims,
            [](const auto& res) { return res.size(); });

    result->do_allocation();

    for (size_t q = 0; q < n; ++q) {
        size_t ofs = result->lims[q];
        for (const auto& [id, distance] : all_results[q]) {
            result->labels[ofs] = id;
            result->distances[ofs] = distance;
            ofs++;
        }
    }

    impl->set_search_parameters(old_sp);
    return;
}

size_t IndexSVSVamana::remove_ids(const IDSelector& sel) {
    std::vector<size_t> ids;
    for (idx_t i = 0; i < ntotal; ++i) {
        if (sel.is_member(i)) {
            ids.emplace_back(i);
        }
    }

    // SVS deletion is a soft deletion, meaning the corresponding vectors are
    // marked as deleted but still present in both the dataset and the graph,
    // and will be navigated through during search.
    // Actual cleanup happens once a large enough number of soft deleted vectors
    // are collected.
    impl->delete_points(ids);
    ntotal -= ids.size();
    ntotal_soft_deleted += ids.size();

    const float cleanup_threshold = .5f;
    if (ntotal == 0 ||
        (float)ntotal_soft_deleted / ntotal > cleanup_threshold) {
        impl->consolidate();
        impl->compact();
        ntotal_soft_deleted = 0;
    }
    return ids.size();
}

void IndexSVSVamana::init_impl(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(!impl);
    FAISS_THROW_IF_NOT(ntotal == 0);

    ntotal = n;
    impl = std::visit(
            [&](auto element) {
                using ElementType = std::decay_t<decltype(element)>;
                return init_impl_t<ElementType>(this, metric_type, n, x);
            },
            get_storage_variant(storage_kind));
}

void IndexSVSVamana::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS index not initialized.");

    // Write index to temporary files and concatenate the contents
    svs_io::SVSTempDirectory tmp;
    impl->save(tmp.config, tmp.graph, tmp.data);
    tmp.write_files_to_stream(out);
}

void IndexSVSVamana::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(
            impl, "Cannot deserialize: SVS index already initialized.");

    // Write stream to files that can be read by DynamicVamana::assemble()
    svs_io::SVSTempDirectory tmp;
    tmp.write_stream_to_files(in);
    impl = std::visit(
            [&](auto element) {
                using ElementType = std::decay_t<decltype(element)>;
                return deserialize_impl_t<ElementType>(tmp, metric_type);
            },
            get_storage_variant(storage_kind));
}

} // namespace faiss

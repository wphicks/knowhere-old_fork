/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CAGRA_CUH
#define CAGRA_CUH

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

#include "common/raft_metric.h"
#include "index/ivf_raft/ivf_raft_config.h"
#include "knowhere/comp/index_param.h"
#include "knowhere/device_bitset.h"
#include "knowhere/factory.h"
#include "knowhere/index_node_thread_pool_wrapper.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"
#include "raft/core/device_resources.hpp"
#include "raft/neighbors/ivf_flat.cuh"
#include "raft/neighbors/ivf_flat_types.hpp"
#include "raft/neighbors/ivf_pq.cuh"
#include "raft/neighbors/ivf_pq_types.hpp"
#include "set_pool.h"
#include "thrust/execution_policy.h"
#include "thrust/logical.h"
#include "thrust/sequence.h"

#ifdef RAFT_COMPILED
#include <raft/neighbors/specializations.cuh>
#endif

namespace knowhere {

namespace detail {
using cagra_index = raft::neighbors::experimental::cagra::index<float, std::int64_t>;

}  // namespace detail

template <typename T>
struct KnowhereConfigType {};

template <>
struct KnowhereConfigType<detail::cagra_index> {
    typedef CagraConfig Type;
};

class CagraIndexNode : public IndexNode {
 public:
    CagraIndexNode(const Object& object) : devs_{}, gpu_index_{} {
    }

    virtual Status
    Build(const DataSet& dataset, const Config& cfg) override {
        auto err = Train(dataset, cfg);
        if (err != Status::success)
            return err;
        return Add(dataset, cfg);
    }

    virtual Status
    Train(const DataSet& dataset, const Config& cfg) override {
        auto cagra_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);
        if (gpu_index_) {
            LOG_KNOWHERE_WARNING_ << "index is already trained";
            return Status::index_already_trained;
        } else if (cagra_cfg.gpu_ids.size() == 1) {
            try {
                auto metric = Str2RaftMetricType(cagra_cfg.metric_type);
                if (!metric.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "please check metric value: " << cagra_cfg.metric_type;
                    return metric.error();
                }
                if (metric.value() != raft::distance::DistanceType::L2Expanded) {
                    LOG_KNOWHERE_WARNING_ << "selected metric not supported in CAGRA indexes: "
                                          << cagra_cfg.metric_type;
                    return Status::invalid_metric_type;
                }
                devs_.insert(devs_.begin(), cagra_cfg.gpu_ids.begin(), cagra_cfg.gpu_ids.end());
                auto scoped_device = detail::device_setter{*cagra_cfg.gpu_ids.begin()};
                auto& res_ = raft_util::get_raft_resources();
                auto rows = dataset.GetRows();
                auto dim = dataset.GetDim();
                auto* data = reinterpret_cast<float const*>(dataset.GetTensor());

                auto data_gpu = raft::make_device_matrix<float, std::int64_t>(res_, rows, dim);
                raft::copy(res_, data_gpu.data_handle(), data, data_gpu.size());

                // TODO(wphicks): set up build params for CAGRA
                auto build_params = raft::neighbors::ivf_flat::index_params{};
                build_params.metric = metric.value();
                build_params.n_lists = ivf_raft_cfg.nlist;
                build_params.kmeans_n_iters = ivf_raft_cfg.kmeans_n_iters;
                build_params.kmeans_trainset_fraction = ivf_raft_cfg.kmeans_trainset_fraction;
                build_params.adaptive_centers = ivf_raft_cfg.adaptive_centers;
                gpu_index_ = raft::neighbors::ivf_flat::build<float, std::int64_t>(res_, build_params, data_gpu.view());

                dim_ = dim;
                counts_ = rows;
                res_.sync_stream();

            } catch (std::exception& e) {
                LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
                return Status::raft_inner_error;
            }
        } else {
            LOG_KNOWHERE_WARNING_ << "RAFT IVF implementation is single-GPU only";
            return Status::raft_inner_error;
        }
        return Status::success;
    }

    virtual Status
    Add(const DataSet& dataset, const Config& cfg) override {
        auto result = Status::success;
        if (!gpu_index_) {
            result = Status::index_not_trained;
        } else {
            try {
                auto rows = dataset.GetRows();
                auto dim = dataset.GetDim();
                auto* data = reinterpret_cast<float const*>(dataset.GetTensor());
                auto scoped_device = detail::device_setter{devs_[0]};
                auto* res_ = &raft_res_pool::get_context().resources_;

                auto stream = res_->get_stream();
                // TODO(wphicks): Clean up transfer with raft
                // buffer objects when available
                auto data_gpu = raft::make_device_matrix<float, std::int64_t>(*res_, rows, dim);
                RAFT_CUDA_TRY(cudaMemcpyAsync(data_gpu.data_handle(), data, data_gpu.size() * sizeof(float),
                                              cudaMemcpyDefault, stream.value()));

                auto indices = rmm::device_uvector<std::int64_t>(rows, stream);
                thrust::sequence(res_->get_thrust_policy(), indices.begin(), indices.end(), gpu_index_->size());

                if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
                    raft::neighbors::ivf_flat::extend<float, std::int64_t>(
                        *res_, raft::make_const_mdspan(data_gpu.view()),
                        std::make_optional(
                            raft::make_device_vector_view<const std::int64_t, std::int64_t>(indices.data(), rows)),
                        gpu_index_.value());
                } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
                    raft::neighbors::ivf_pq::extend<float, std::int64_t>(
                        *res_, raft::make_const_mdspan(data_gpu.view()),
                        std::make_optional(
                            raft::make_device_matrix_view<const std::int64_t, std::int64_t>(indices.data(), rows, 1)),
                        gpu_index_.value());
                } else {
                    static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
                }
                dim_ = dim;
                counts_ = rows;
            } catch (std::exception& e) {
                LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
                result = Status::raft_inner_error;
            }
        }

        return result;
    }

    virtual expected<DataSetPtr, Status>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        auto ivf_raft_cfg = static_cast<const typename KnowhereConfigType<T>::Type&>(cfg);
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto* data = reinterpret_cast<float const*>(dataset.GetTensor());
        auto output_size = rows * ivf_raft_cfg.k;
        auto ids = std::unique_ptr<std::int64_t[]>(new std::int64_t[output_size]);
        auto dis = std::unique_ptr<float[]>(new float[output_size]);
        try {
            auto scoped_device = detail::device_setter{devs_[0]};
            auto* res_ = &raft_res_pool::get_context().resources_;

            // TODO(wphicks): Clean up transfer with raft
            // buffer objects when available
            auto data_gpu = raft::make_device_matrix<float, std::int64_t>(*res_, rows, dim);
            raft::copy(data_gpu.data_handle(), data, data_gpu.size(), res_->get_stream());

            auto gpu_results = raft_detail::raft_results{*res_};
            auto gpu_bitset = DeviceBitset{*res_, bitset};

            if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
                auto search_params = raft::neighbors::ivf_flat::search_params{};
                search_params.n_probes = std::min<uint32_t>(ivf_raft_cfg.nprobe, gpu_index_->n_lists());
                auto search_k = std::min(
                    ivf_raft_cfg.k + (bitset.count() * ivf_raft_cfg.k / counts_),
                    std::min(static_cast<uint64_t>(counts_), static_cast<uint64_t>(raft_detail::MAX_IVF_FLAT_K)));
                gpu_results = RawSearch(*res_, raft::make_const_mdspan(data_gpu.view()), search_params, search_k,
                                        ivf_raft_cfg.k, gpu_bitset.view());
            } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
                auto search_params = raft::neighbors::ivf_pq::search_params{};
                search_params.n_probes = std::min<uint32_t>(ivf_raft_cfg.nprobe, gpu_index_->n_lists());

                auto lut_dtype = detail::str_to_cuda_dtype(ivf_raft_cfg.lut_dtype);
                if (!lut_dtype.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "please check lookup dtype: " << ivf_raft_cfg.lut_dtype;
                    return unexpected(lut_dtype.error());
                }
                if (lut_dtype.value() != CUDA_R_32F && lut_dtype.value() != CUDA_R_16F &&
                    lut_dtype.value() != CUDA_R_8U) {
                    LOG_KNOWHERE_WARNING_ << "selected lookup dtype not supported: " << ivf_raft_cfg.lut_dtype;
                    return unexpected(Status::invalid_args);
                }
                search_params.lut_dtype = lut_dtype.value();
                auto internal_distance_dtype = detail::str_to_cuda_dtype(ivf_raft_cfg.internal_distance_dtype);
                if (!internal_distance_dtype.has_value()) {
                    LOG_KNOWHERE_WARNING_ << "please check internal distance dtype: "
                                          << ivf_raft_cfg.internal_distance_dtype;
                    return unexpected(internal_distance_dtype.error());
                }
                if (internal_distance_dtype.value() != CUDA_R_32F && internal_distance_dtype.value() != CUDA_R_16F) {
                    LOG_KNOWHERE_WARNING_ << "selected internal distance dtype not supported: "
                                          << ivf_raft_cfg.internal_distance_dtype;
                    return unexpected(Status::invalid_args);
                }
                search_params.internal_distance_dtype = internal_distance_dtype.value();
                search_params.preferred_shmem_carveout = search_params.preferred_shmem_carveout;
                gpu_results = RawSearch(*res_, raft::make_const_mdspan(data_gpu.view()), search_params, ivf_raft_cfg.k,
                                        ivf_raft_cfg.k, gpu_bitset.view());
            } else {
                static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
            }

            if (gpu_results.k() != ivf_raft_cfg.k) {
                auto new_gpu_results = raft_detail::raft_results{*res_, gpu_results.rows(), ivf_raft_cfg.k};
                raft_detail::slice<<<1024, 256, 0, res_->get_stream().value()>>>(
                    new_gpu_results.ids_data(), gpu_results.ids_data(), new_gpu_results.rows(), new_gpu_results.k(),
                    gpu_results.rows(), gpu_results.k());
                res_->sync_stream();
                gpu_results = new_gpu_results;
            }
            raft::copy(ids.get(), gpu_results.ids_data(), output_size, res_->get_stream());
            raft::copy(dis.get(), gpu_results.dists_data(), output_size, res_->get_stream());
            res_->sync_stream();

        } catch (std::exception& e) {
            LOG_KNOWHERE_WARNING_ << "RAFT inner error, " << e.what();
            return unexpected(Status::raft_inner_error);
        }

        return GenResultDataSet(rows, ivf_raft_cfg.k, ids.release(), dis.release());
    }

    expected<DataSetPtr, Status>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        return unexpected(Status::not_implemented);
    }

    virtual expected<DataSetPtr, Status>
    GetVectorByIds(const DataSet& dataset) const override {
        return unexpected(Status::not_implemented);
    }

    virtual bool
    HasRawData(const std::string& metric_type) const override {
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            return !IsMetricType(metric_type, metric::COSINE);
        }
        if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            return false;
        }
    }

    expected<DataSetPtr, Status>
    GetIndexMeta(const Config& cfg) const override {
        return unexpected(Status::not_implemented);
    }

    virtual Status
    Serialize(BinarySet& binset) const override {
        if (!gpu_index_.has_value()) {
            LOG_KNOWHERE_ERROR_ << "Can not serialize empty CagraIndex.";
            return Status::empty_index;
        }
        std::stringbuf buf;

        std::ostream os(&buf);

        os.write((char*)(&this->dim_), sizeof(this->dim_));
        os.write((char*)(&this->counts_), sizeof(this->counts_));
        os.write((char*)(&this->devs_[0]), sizeof(this->devs_[0]));

        auto scoped_device = detail::device_setter{devs_[0]};
        auto* res_ = &raft_res_pool::get_context().resources_;

        if constexpr (std::is_same_v<T, detail::raft_ivf_flat_index>) {
            raft::neighbors::ivf_flat::serialize<float, std::int64_t>(*res_, os, *gpu_index_);
        }
        if constexpr (std::is_same_v<T, detail::raft_ivf_pq_index>) {
            raft::neighbors::ivf_pq::serialize<std::int64_t>(*res_, os, *gpu_index_);
        }

        os.flush();
        std::shared_ptr<uint8_t[]> index_binary(new (std::nothrow) uint8_t[buf.str().size()]);

        memcpy(index_binary.get(), buf.str().c_str(), buf.str().size());
        binset.Append(this->Type(), index_binary, buf.str().size());
        return Status::success;
    }

    virtual Status
    Deserialize(const BinarySet& binset, const Config& config) override {
        std::stringbuf buf;
        auto binary = binset.GetByName(this->Type());
        if (binary == nullptr) {
            LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
            return Status::invalid_binary_set;
        }
        buf.sputn((char*)binary->data.get(), binary->size);
        std::istream is(&buf);

        is.read((char*)(&this->dim_), sizeof(this->dim_));
        is.read((char*)(&this->counts_), sizeof(this->counts_));
        this->devs_.resize(1);
        is.read((char*)(&this->devs_[0]), sizeof(this->devs_[0]));
        auto scoped_device = detail::device_setter{devs_[0]};

        raft_res_pool::resource::instance().init(rmm::cuda_device_id(devs_[0]));
        auto* res_ = &raft_res_pool::get_context().resources_;

        if constexpr (std::is_same_v<T, detail::raft_ivf_flat_index>) {
            T index_ = raft::neighbors::ivf_flat::deserialize<float, std::int64_t>(*res_, is);
            is.sync();
            gpu_index_ = T(std::move(index_));
        }
        if constexpr (std::is_same_v<T, detail::raft_ivf_pq_index>) {
            T index_ = raft::neighbors::ivf_pq::deserialize<std::int64_t>(*res_, is);
            is.sync();
            gpu_index_ = T(std::move(index_));
        }
        // TODO(yusheng.ma):support no raw data mode
        /*
    #define RAW_DATA "RAW_DATA"
    auto data = binset.GetByName(RAW_DATA);
    raft_gpu::raw_data_copy(*this->index_, data->data.get(), data->size);
    */
        is.sync();

        return Status::success;
    }

    virtual Status
    DeserializeFromFile(const std::string& filename, const Config& config) {
        LOG_KNOWHERE_ERROR_ << "CagraIndex doesn't support Deserialization from file.";
        return Status::not_implemented;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<typename KnowhereConfigType<T>::Type>();
    }

    virtual int64_t
    Dim() const override {
        return dim_;
    }

    virtual int64_t
    Size() const override {
        return 0;
    }

    virtual int64_t
    Count() const override {
        return counts_;
    }

    virtual std::string
    Type() const override {
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            return knowhere::IndexEnum::INDEX_RAFT_IVFFLAT;
        }
        if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            return knowhere::IndexEnum::INDEX_RAFT_IVFPQ;
        }
    }

 private:
    std::vector<int32_t> devs_;
    int64_t dim_ = 0;
    int64_t counts_ = 0;
    std::optional<T> gpu_index_;

    template <typename raft_search_params_t>
    raft_detail::raft_results
    RawSearch(raft::device_resources& res, raft::device_matrix_view<const float, std::int64_t> queries,
              raft_search_params_t const& search_params, int k, int target_k, DeviceBitsetView const& bitset) const {
        auto max_k = counts_;
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            k = std::min(k, raft_detail::MAX_IVF_FLAT_K);
            max_k = std::min(max_k, int64_t{raft_detail::MAX_IVF_FLAT_K});
        }
        auto result = raft_detail::raft_results{res, queries.extent(0), k};
        if constexpr (std::is_same_v<detail::raft_ivf_flat_index, T>) {
            raft::neighbors::ivf_flat::search<float, std::int64_t>(res, search_params, *gpu_index_, queries,
                                                                   result.ids(), result.dists());
        } else if constexpr (std::is_same_v<detail::raft_ivf_pq_index, T>) {
            raft::neighbors::ivf_pq::search<float, std::int64_t>(res, search_params, *gpu_index_, queries, result.ids(),
                                                                 result.dists());
        } else {
            static_assert(std::is_same_v<detail::raft_ivf_flat_index, T>);
        }

        auto blocks = std::min(queries.extent(0), int64_t{1024});
        auto threads = std::min(k, 256);
        auto warp_remainder = threads % 32;
        if (warp_remainder != 0) {
            threads += (32 - warp_remainder);
        }
        auto enough_valid = raft::make_device_vector<bool>(res, queries.extent(0));

        raft_detail::postprocess_device_results<<<blocks, threads, 0, res.get_stream().value()>>>(
            enough_valid.data_handle(), result.ids_data(), result.dists_data(), queries.extent(0), k, target_k, bitset);

        if (k < max_k && !thrust::all_of(res.get_thrust_policy(), enough_valid.data_handle(),
                                         enough_valid.data_handle() + queries.extent(0), thrust::identity<bool>())) {
            result = RawSearch(res, queries, search_params, std::min(int64_t{k * 2}, max_k), target_k, bitset);
        }
        return result;
    }
};

void
SetRaftMemPool(size_t init_size, size_t max_size) {
    LOG_KNOWHERE_INFO_ << "Set GPU pool size: init size " << init_size << ", max size " << max_size;
    raft_res_pool::resource::instance().set_pool_size(init_size, max_size);
}

}  // namespace knowhere
#endif /* CAGRA_CUH */

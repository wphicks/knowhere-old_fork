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

#include <thrust/optional.h>

#include <optional>
#include <rmm/cuda_stream_view.hpp>

#ifndef RAFT_UTILS_CUH
#define RAFT_UTILS_CUH

namespace knowhere {
namespace raft_util {

/* A convenience struct for holding output from a RAFT index search */
struct raft_results {
    raft_results(raft::device_resources& res)
        : ids_{raft::make_device_matrix<std::int64_t, std::int64_t>(res, 0, 0)},
          dists_{raft::make_device_matrix<float, std::int64_t>(res, 0, 0)} {
    }
    raft_results(raft::device_resources& res, std::int64_t rows, std::int64_t k)
        : ids_{raft::make_device_matrix<std::int64_t, std::int64_t>(res, rows, k)},
          dists_{raft::make_device_matrix<float, std::int64_t>(res, rows, k)} {
    }
    auto
    ids() {
        return ids_.view();
    }
    auto
    dists() {
        return dists_.view();
    }
    auto
    ids_data() {
        return ids_.data_handle();
    }
    auto
    dists_data() {
        return dists_.data_handle();
    }

    auto
    rows() const {
        return ids_.extent(0);
    }
    auto
    k() const {
        return ids_.extent(1);
    }

 private:
    raft::device_matrix<std::int64_t, std::int64_t> ids_;
    raft::device_matrix<float, std::int64_t> dists_;
};

/* The following kernel is used to filter results from a RAFT search based on
 * a bitset and ensure that invalid IDs are represented correctly for
 * knowhere conventions */
inline __global__ void
postprocess_device_results(bool* enough_valid, int64_t* ids, float* dists, int64_t rows, int k, int target_k,
                           DeviceBitsetView bitset) {
    __shared__ int invalid_count[1];
    for (auto row_index = blockIdx.x; row_index < rows; row_index += gridDim.x) {
        if (threadIdx.x == 0) {
            invalid_count[0] = 0;
        }
        __syncthreads();
        // First, replace all invalid IDs with -1
        for (auto col_index = threadIdx.x; col_index < k; col_index += blockDim.x) {
            auto elem_index = row_index * k + col_index;
            auto cur_id = ids[elem_index];
            auto invalid_id = bitset.test(cur_id);
            if (invalid_id) {  // TODO(wphicks): assuming the branch is worth it here;
                               // should analyze perf.
                atomicAdd_block(invalid_count, 1);
            }
            invalid_id |= cur_id == std::numeric_limits<int64_t>::max();
            ids[elem_index] = int(invalid_id) * -1 + int(!invalid_id) * ids[elem_index];
        }
        __syncthreads();

        // Now move all valid results to the front of the row
        auto cur_valid_index = row_index * k;
        for (auto col_index = 0; col_index < k; ++col_index) {
            auto elem_index = row_index * k + col_index;
            // Just do one row per block for now; can improve this later
            if (ids[elem_index] != -1 && threadIdx.x == 0) {
                // Swap valid id to an earlier place in the row
                ids[cur_valid_index] = ids[elem_index];
                if (elem_index != cur_valid_index) {
                    ids[elem_index] = -1;
                    // Only count elements invalidated by the bitset. These are the
                    // elements that will require a swap as opposed to just appearing at
                    // the end of the row
                }
                dists[cur_valid_index++] = dists[elem_index];
            }
        }
        // Check if we have enough valid results
        if (threadIdx.x == 0) {
            enough_valid[row_index] = (k - invalid_count[0] >= target_k);
        }
    }
}

/* The following kernel is used to copy data from one row-major 2D array to
 * another. If the input array has more columns than the output array, the
 * rows will be truncated to the length of the output array. Similarly, the
 * number of rows will be truncated if the output has fewer rows. If the output
 * array has more rows/columns than the input, the output will be padded with
 * entries of -1. */
template <typename T, typename IndexT>
__global__ void
slice(T* out, T* in, IndexT out_rows, IndexT out_cols, IndexT in_rows, IndexT in_cols) {
    for (auto i = blockIdx.x * blockDim.x + threadIdx.x; i < out_rows * out_cols; i += blockDim.x * gridDim.x) {
        auto row_index = i / out_cols;
        auto col_index = i % out_cols;
        if (row_index < in_rows && col_index < in_cols) {
            out[i] = in[row_index * in_cols + col_index];
        } else {
            out[i] = -1;
        }
    }
}

// TODO(wphicks): Replace this with version from RAFT once merged
struct device_setter {
    device_setter(int new_device)
        : prev_device_{[]() {
              auto result = int{};
              RAFT_CUDA_TRY(cudaGetDevice(&result));
              return result;
          }()} {
        RAFT_CUDA_TRY(cudaSetDevice(new_device));
    }

    ~device_setter() {
        RAFT_CUDA_TRY_NO_THROW(cudaSetDevice(prev_device_));
    }

 private:
    int prev_device_;
};

template <typename key_t, typename mapped_t>
struct fastmap {
    using key_type = key_t;
    using mapped_type = value_t;
    using value_type = std::pair<key_type, mapped_type>;
    fastmap() : data_{} {
    }
    fastmap(std::initializer_list<value_type> init)
        : data_{[](auto&& init_) {
              auto result = std::vector{std::move(init_)};
              std::sort(result.begin(), result.end(),
                        [](auto&& entry1, auto&& entry2) { return entry1.first < entry2.first; });
              return result;
          }(std::move(init))} {
    }
    void
    set(key_type key, mapped_type&& value) {
        data_.insert(
            std::upper_bound(data_.begin(), data_.end(), key,
                             [](key_type const& new_key, value_type const& entry) { return new_key < entry.first; }),
            std::make_pair(std::move(key), std::move(value)));
    }
    auto
    check(key_type key) {
        auto iter = search(key);
        return iter != data_.end() && iter->first == key;
    }

    auto
    get(key_type key) {
        auto result = std::optional<std::reference_wrapper<mapped_type const>>{};
        auto iter = search(key);
        if (iter != data_.end() && iter->first == key) {
            result = iter->second;
        }
        return result;
    }

 private:
    std::vector<value_type> data_;
    auto
    search(key_type key) {
        return std::lower_bound(
            data_.begin(), data_.end(), key,
            [](value_type const& entry, key_type const& lookup_key) { return entry.first < lookup_key; });
    }
};

namespace detail {
struct context {
    context() : resources_{rmm::cuda_stream_per_thread.view(), nullptr, rmm::mr::get_current_device_resource()} {
    }
    ~context() = default;
    context(context&&) = delete;
    context(context const&) = delete;
    context&
    operator=(context&&) = delete;
    context&
    operator=(context const&) = delete;
    raft::device_resources resources_;
};

inline thread_local auto raft_context = context{};

}  // namespace detail

inline auto&
get_raft_resources() {
    return detail::raft_context.resources_;
}

class resource {
 public:
    static resource&
    instance() {
        static resource res;
        return res;
    }

    void
    set_pool_size(std::optional<std::size_t> init_size = std::nullopt,
                  std::optional<std::size_t> max_size = std::nullopt) {
        if (init_size.has_value()) {
            initial_pool_size = thrust::make_optional(init_size);
        } else {
            initial_pool_size = thrust::nullopt;
        }
        if (max_size.has_value()) {
            maximum_pool_size = thrust::make_optional(max_size);
        } else {
            maximum_pool_size = thrust::nullopt;
        }
    }

    void
    init(rmm::cuda_device_id device_id) {
        std::lock_guard<std::mutex> lock(mtx_);
        if (!map_.check(device_id.value()) {
            char* env_str = getenv("KNOWHERE_GPU_MEM_POOL_SIZE");
            if (env_str != NULL) {
                auto initial_pool_size_tmp = std::size_t{};
                auto maximum_pool_size_tmp = std::size_t{};
                auto stat = sscanf(env_str, "%zu;%zu", &initial_pool_size_tmp, &maximum_pool_size_tmp);
                if (stat == 2) {
                    LOG_KNOWHERE_INFO_ << "Get Gpu Pool Size From env, init size: " << initial_pool_size_tmp
                                       << " MB, max size: " << maximum_pool_size_tmp << " MB";
                    initial_pool_size = initial_pool_size_tmp << 20;
                    maximum_pool_size = maximum_pool_size_tmp << 20;
                } else {
                    LOG_KNOWHERE_WARNING_ << "please check env format";
                }
            }

            auto mr_ = std::make_unique<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>(
                rmm::initial_resource(), initial_pool_size, maximum_pool_size);
            rmm::mr::set_per_device_resource(device_id, mr_.get());
            map_.set(device_id.value(), std::move(mr_));
        }
    }

 private:
    resource(){};
    ~resource(){};
    resource(resource&&) = delete;
    resource(resource const&) = delete;
    resource&
    operator=(resource&&) = delete;
    resource&
    operator=(resource const&) = delete;
    fastmap<rmm::cuda_device_id::value_type,
            std::unique_ptr<rmm::mr::pool_memory_resource<rmm::mr::cuda_memory_resource>>>
        map_;
    mutable std::mutex mtx_;
    thrust::optional<std::size_t> initial_pool_size = thrust::nullopt;
    thrust::optional<std::size_t> maximum_pool_size = thrust::nullopt;
};

namespace cuda_type {
auto static constexpr const CUDA_R_16F = "CUDA_R_16F";
auto static constexpr const CUDA_C_16F = "CUDA_C_16F";
auto static constexpr const CUDA_R_16BF = "CUDA_R_16BF";
auto static constexpr const CUDA_C_16BF = "CUDA_C_16BF";
auto static constexpr const CUDA_R_32F = "CUDA_R_32F";
auto static constexpr const CUDA_C_32F = "CUDA_C_32F";
auto static constexpr const CUDA_R_64F = "CUDA_R_64F";
auto static constexpr const CUDA_C_64F = "CUDA_C_64F";
auto static constexpr const CUDA_R_8I = "CUDA_R_8I";
auto static constexpr const CUDA_C_8I = "CUDA_C_8I";
auto static constexpr const CUDA_R_8U = "CUDA_R_8U";
auto static constexpr const CUDA_C_8U = "CUDA_C_8U";
auto static constexpr const CUDA_R_32I = "CUDA_R_32I";
auto static constexpr const CUDA_C_32I = "CUDA_C_32I";
auto static constexpr const CUDA_R_8F_E4M3 = "CUDA_R_8F_E4M3";
auto static constexpr const CUDA_R_8F_E5M2 = "CUDA_R_8F_E5M2";
}  // namespace cuda_type

inline auto
str_to_cuda_dtype(std::string const& str) {
    static const name_map = fastmap<std::string, cudaDataType_t>{
        {cuda_type::CUDA_R_16F, CUDA_R_16F},   {cuda_type::CUDA_C_16F, CUDA_C_16F},
        {cuda_type::CUDA_R_16BF, CUDA_R_16BF}, {cuda_type::CUDA_C_16BF, CUDA_C_16BF},
        {cuda_type::CUDA_R_32F, CUDA_R_32F},   {cuda_type::CUDA_C_32F, CUDA_C_32F},
        {cuda_type::CUDA_R_64F, CUDA_R_64F},   {cuda_type::CUDA_C_64F, CUDA_C_64F},
        {cuda_type::CUDA_R_8I, CUDA_R_8I},     {cuda_type::CUDA_C_8I, CUDA_C_8I},
        {cuda_type::CUDA_R_8U, CUDA_R_8U},     {cuda_type::CUDA_C_8U, CUDA_C_8U},
        {cuda_type::CUDA_R_32I, CUDA_R_32I},   {cuda_type::CUDA_C_32I, CUDA_C_32I},
        // not support, when we use cuda 11.6
        //{cuda_type::CUDA_R_8F_E4M3, CUDA_R_8F_E4M3}, {cuda_type::CUDA_R_8F_E5M2, CUDA_R_8F_E5M2},

    };

    return name_map.get(str);
}

}  // namespace raft_util
}  // namespace knowhere
#endif /* RAFT_UTILS_CUH */

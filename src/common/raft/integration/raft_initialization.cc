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
#include <cuda_runtime_api.h>
#include <raft/core/device_resources_manager.hpp>
#include <raft/core/device_setter.hpp>
#include <raft/core/resource/device_memory_resource.hpp>
#include "common/raft/integration/raft_initialization.hpp"
namespace raft_knowhere {

void initialize_raft(raft_configuration const& config) {
  raft::device_resources_manager::set_streams_per_device(config.streams_per_device);
  if (config.stream_pool_size) {
    raft::device_resources_manager::set_stream_pools_per_device(
      config.stream_pools_per_device,
      *(config.stream_pool_size)
    );
  } else {
    raft::device_resources_manager::set_stream_pools_per_device(config.stream_pools_per_device);
  }
  if (config.init_mem_pool_size_mb && config.max_mem_pool_size_mb.value_or(1) > 0) {
    raft::device_resources_manager::set_init_mem_pool_size(*(config.init_mem_pool_size_mb) << 20);
  }
  if (config.max_mem_pool_size_mb.value_or(0) > 0) {
    raft::device_resources_manager::set_max_mem_pool_size(*(config.max_mem_pool_size_mb) << 20);
  }
  if (config.max_workspace_size_mb) {
    raft::device_resources_manager::set_workspace_allocation_limit(*(config.max_workspace_size_mb) << 20);
  }
  auto device_count = []() {
    auto result = 0;
    RAFT_CUDA_TRY(cudaGetDeviceCount(&result));
    RAFT_EXPECTS(result != 0, "No CUDA devices found");
    return result;
  }();

  if (config.max_workspace_size_mb) {
    auto workspace_size = *(config.max_workspace_size_mb) << 20;
    for (auto device_id = 0; device_id < device_count; ++device_id) {
      auto scoped_device = raft::device_setter{device_id};
      raft::device_resources_manager::set_workspace_memory_resource(
        raft::resource::workspace_resource_factory::default_pool_resource(
          workspace_size
        ),
        device_id
      );
    }
  }
}

} // namespace raft_knowhere

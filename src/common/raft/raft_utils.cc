#include "raft_utils.h"

#include <cstddef>
#include <mutex>
#include <optional>
#include <raft/core/device_resources.hpp>
#include <raft/core/device_resources_manager.hpp>
#include <sstream>

#include "knowhere/log.h"

namespace raft_utils {
void
init_gpu_resources(std::optional<std::size_t> streams_per_device) {
    static std::once_flag flag;
    std::call_once(flag, [streams_per_device]() {
        auto stream_count = streams_per_device.value_or([]() {
            auto result = std::size_t{16};
            if (auto* env_str = std::getenv("KNOWHERE_STREAMS_PER_GPU")) {
                auto str_stream = std::stringstream{env_str};
                str_stream >> result;
                if (str_stream.fail() || result == std::size_t{0}) {
                    LOG_KNOWHERE_WARNING_ << "KNOWHERE_STREAMS_PER_GPU env variable should be a positive integer";
                    result = std::size_t{16};
                } else {
                    LOG_KNOWHERE_INFO_ << "streams per gpu set to " << result;
                }
            }
            return result;
        }());
        raft::device_resources_manager::set_streams_per_device(stream_count);

        auto* env_str = getenv("KNOWHERE_GPU_MEM_POOL_SIZE");
        if (env_str != NULL) {
            auto init_pool_size_tmp = std::size_t{};
            auto max_pool_size_tmp = std::size_t{};
            auto stat = sscanf(env_str, "%zu;%zu", &init_pool_size_tmp, &max_pool_size_tmp);
            if (stat == 2) {
                LOG_KNOWHERE_INFO_ << "Get Gpu Pool Size From env, init size: " << init_pool_size_tmp
                                   << " MB, max size: " << max_pool_size_tmp << " MB";
                raft::device_resources_manager::set_mem_pool(init_pool_size_tmp << 20, max_pool_size_tmp << 20);
            } else {
                LOG_KNOWHERE_WARNING_ << "please check env format";
            }
        }
    });
}

int
gpu_device_manager::random_choose() const {
    srand(time(NULL));
    return rand() % memory_load_.size();
}

int
gpu_device_manager::choose_with_load(size_t load) {
    std::lock_guard<std::mutex> lock(mtx_);

    auto it = std::min_element(memory_load_.begin(), memory_load_.end());
    *it += load;
    return std::distance(memory_load_.begin(), it);
}

gpu_device_manager::gpu_device_manager() {
    int device_counts;
    try {
        RAFT_CUDA_TRY(cudaGetDeviceCount(&device_counts));
    } catch (const raft::exception& e) {
        LOG_KNOWHERE_FATAL_ << e.what();
    }
    memory_load_.resize(device_counts);
    std::fill(memory_load_.begin(), memory_load_.end(), 0);
}

gpu_device_manager&
gpu_device_manager::instance() {
    static gpu_device_manager mgr;
    return mgr;
}

}  // namespace raft_utils

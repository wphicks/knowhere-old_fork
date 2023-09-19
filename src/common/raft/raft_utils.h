#include <cstddef>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <vector>

#include "knowhere/log.h"
#include "raft/core/device_resources.hpp"
#include "raft/core/device_resources_manager.hpp"

namespace raft_utils {

inline void
init_gpu_resources(std::optional<std::size_t> streams_per_device = std::nullopt) {
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
        raft::device_memory_resources::set_streams_per_device(stream_count);

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

class gpu_device_manager {
 public:
    static gpu_device_manager&
    instance();
    int
    random_choose() const;
    int
    choose_with_load(size_t load);

 private:
    gpu_device_manager();
    std::vector<size_t> memory_load_;
    mutable std::mutex mtx_;
};

};  // namespace raft_utils

#define RANDOM_CHOOSE_DEVICE_WITH_ASSIGN(x)                             \
    do {                                                                \
        x = raft_utils::gpu_device_manager::instance().random_choose(); \
    } while (0)
#define MIN_LOAD_CHOOSE_DEVICE_WITH_ASSIGN(x, load)                            \
    do {                                                                       \
        x = raft_utils::gpu_device_manager::instance().choose_with_load(load); \
    } while (0)

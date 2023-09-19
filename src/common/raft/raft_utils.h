#include <cstddef>
#include <mutex>
#include <optional>
#include <vector>

namespace raft_utils {

void
init_gpu_resources(std::optional<std::size_t> streams_per_device = std::nullopt);

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

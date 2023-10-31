#include <cstddef>
#include <optional>
namespace raft_knowhere {
struct raft_configuration {
  std::size_t streams_per_device = std::size_t{16};
  std::size_t stream_pools_per_device = std::size_t{};
  std::optional<std::size_t> stream_pool_size = std::nullopt;
  std::optional<std::size_t> init_mem_pool_size_mb = std::nullopt;
  std::optional<std::size_t> max_mem_pool_size_mb = std::nullopt;
  std::optional<std::size_t> max_workspace_size_mb = std::nullopt;
};

void initialize_raft(raft_configuration const& config);
}  // namespace raft_knowhere

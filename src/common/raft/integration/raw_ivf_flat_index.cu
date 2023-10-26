#include "common/raft/proto/raft_index.cuh"
namespace raft_proto {
template struct raft_index<raft::neighbors::ivf_flat::index, float, std::int64_t>;
}

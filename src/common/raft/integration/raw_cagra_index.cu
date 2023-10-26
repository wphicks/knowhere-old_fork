#include "common/raft/proto/raft_index.cuh"
namespace raft_proto {
template struct raft_index<raft::neighbors::cagra::index, float, std::uint32_t>;
}

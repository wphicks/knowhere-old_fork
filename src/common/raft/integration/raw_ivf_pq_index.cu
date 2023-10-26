#include "common/raft/proto/raft_index.cuh"
namespace raft_proto {
template struct raft_index<raft::neighbors::ivf_pq::index, std::int64_t>;
}

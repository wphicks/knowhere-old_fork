#include "common/raft/proto/raft_index_kind.hpp"
#include "common/raft/integration/raft_knowhere_index.cuh"
namespace raft_knowhere {
template struct raft_knowhere_index<raft_proto::raft_index_kind::cagra>;
}  // namespace raft_knowhere

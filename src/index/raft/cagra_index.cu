#include "index/raft/raft_index_kind.hpp"
#include "index/raft/raft_knowhere_index.cuh"
namespace raft_knowhere {
template struct raft_knowhere_index<raft_proto::raft_index_kind::cagra>;
}  // namespace raft_knowhere

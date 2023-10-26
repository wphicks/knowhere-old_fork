#pragma once

namespace raft_proto {
enum struct raft_index_kind {
  ivf_flat,
  ivf_pq,
  cagra
};
}  // namespace raft_proto

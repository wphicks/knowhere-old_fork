#pragma once
namespace raft_proto {

template <typename underlying_index, typename T, typename IdxT>
struct batched_raft_index{
 private:
  struct batched_raft_index_impl {
   private:
    raft::device_matrix<T> search_data_;
    underlying_index index_;
    raft::device_matrix<T> output_data_;
  };
  streamsafe_wrapper<batched_raft_index_impl> impl_;
};

}  // namespace raft_proto

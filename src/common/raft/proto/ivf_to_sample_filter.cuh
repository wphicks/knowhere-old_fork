namespace raft_proto {

template <typename index_t, typename filter_t>
struct ivf_to_sample_filter {
  const index_t* const* inds_ptrs_;
  const filter_t next_filter_;

  ivf_to_sample_filter(const index_t* const* inds_ptrs, const filter_t next_filter)
    : inds_ptrs_{inds_ptrs}, next_filter_{next_filter} {}

  inline __host__ __device__ bool operator()(
    // query index
    const uint32_t query_ix,
    // the current inverted list index
    const uint32_t cluster_ix,
    // the index of the current sample inside the current inverted list
    const uint32_t sample_ix) const
  {
    return next_filter_(query_ix, inds_ptrs_[cluster_ix][sample_ix]);
  }
};

} // namespace raft_proto

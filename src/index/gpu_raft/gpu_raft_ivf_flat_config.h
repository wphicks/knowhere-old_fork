#ifndef GPU_RAFT_IVF_FLAT_CONFIG_H
#define GPU_RAFT_IVF_FLAT_CONFIG_H

#include "knowhere/config.h"
#include "index/ivf/ivf_config.h"
#include "index/raft/raft_index_kind.hpp"
#include "index/raft/raft_knowhere_config.hpp"

namespace knowhere {

struct GpuRaftIvfFlatConfig : public IvfFlatConfig {
    CFG_INT kmeans_n_iters;
    CFG_FLOAT kmeans_trainset_fraction;
    CFG_BOOL adaptive_centers;
    KNOHWERE_DECLARE_CONFIG(GpuRaftIvfFlatConfig) {
        KNOWHERE_CONFIG_DECLARE_FIELD(k)
            .set_default(10)
            .description("search for top k similar vector.")
            .set_range(1, 256)  // Declared in base but limited to 256
            .for_search();
        KNOWHERE_CONFIG_DECLARE_FIELD(kmeans_n_iters)
            .description("iterations to search for kmeans centers")
            .set_default(20)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(kmeans_trainset_fraction)
            .description("fraction of data to use in kmeans building")
            .set_default(0.5)
            .for_train();
        KNOWHERE_CONFIG_DECLARE_FIELD(adaptive_centers)
            .description("update centroids with new data")
            .set_default(false)
            .for_train();
    }
};

[[nodiscard]] inline auto to_raft_knowhere_config(GpuRaftIvfFlatConfig const& cfg) {
  auto result = raft_knowhere::raft_knowhere_config{raft_proto::raft_index_kind::ivf_flat};

  result.metric_type = cfg.metric_type.value();
  result.k = cfg.k.value();
  result.nlist = cfg.nlist;
  result.nprobe = cfg.nprobe;
  result.kmeans_n_iters = cfg.kmeans_n_iters;
  result.kmeans_trainset_fraction = cfg.kmeans_trainset_fraction;
  result.adaptive_centers = cfg.adaptive_centers;

  return result;
}

}  // namespace knowhere

#endif /*GPU_RAFT_IVF_FLAT_CONFIG_H*/

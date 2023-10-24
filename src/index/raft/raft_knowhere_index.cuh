#pragma once
#include <cstdint>
#include <type_traits>
#include <raft/core/device_resources_manager.hpp>
#include <raft/distance/distance_types.hpp>
#include "index/raft/raft_index_kind.hpp"
#include "index/raft/raft_index.cuh"

namespace raft_knowhere {
namespace detail {

// This helper struct maps the generic type of RAFT index to the specific
// instantiation of that index used within knowhere.
template <bool B, raft_index_kind IndexKind>
struct raft_index_type_mapper : std::false_type {
};

template <>
struct raft_index_type_mapper<true, raft_index_kind::ivf_flat> : std::true_type {
  using type = raft_index<raft::neighbors::ivf_flat::index, float, std::int64_t>;
  using data_type = float;
  using underlying_index_type = typename type::vector_index_type;
  using index_params_type = typename type::index_params_type;
  using search_params_type = typename type::search_params_type;
};
template <>
struct raft_index_type_mapper<true, raft_index_kind::ivf_pq> : std::true_type {
  using type = raft_index<raft::neighbors::ivf_pq::index, std::int64_t>;
  using data_type = float;
  using underlying_index_type = typename type::vector_index_type;
  using index_params_type = typename type::index_params_type;
  using search_params_type = typename type::search_params_type;
}; 
template <>
struct raft_index_type_mapper<true, raft_index_kind::cagra> : std::true_type {
  using type = raft_index<raft::neighbors::cagra::index, float, std::uint32_t>;
  using data_type = float;
  using underlying_index_type = typename type::vector_index_type;
  using index_params_type = typename type::index_params_type;
  using search_params_type = typename type::search_params_type;
}; 

} // namespace detail

template <raft_index_kind IndexKind>
using raft_index_t = typename detail::raft_index_type_mapper<true, IndexKind>::type;

template <raft_index_kind IndexKind>
using raft_data_t = typename detail::raft_index_type_mapper<true, IndexKind>::data_t;

template <raft_index_kind IndexKind>
using raft_index_params_t = typename detail::raft_index_type_mapper<true, IndexKind>::index_params_type;
template <raft_index_kind IndexKind>
using raft_search_params_t = typename detail::raft_index_type_mapper<true, IndexKind>::search_params_type;

// Metrics are passed between knowhere and RAFT as strings to avoid tight
// coupling between the implementation details of either one.
[[nodiscard]] inline auto metric_string_to_raft_distance_type(std::string const& metric_string) {
  auto result = raft::distance::DistanceType::L2Expanded;
  if (metric_string == "L2Expanded") {
    result = raft::distance::DistanceType::L2Expanded;
  } else if (metric_string == "L2SqrtExpanded") {
    result = raft::distance::DistanceType::L2SqrtExpanded;
  } else if (metric_string == "CosineExpanded") {
   result = raft::distance::DistanceType::CosineExpanded;
  } else if (metric_string == "L1") {
    result = raft::distance::DistanceType::L1;
  } else if (metric_string == "L2Unexpanded") {
    result = raft::distance::DistanceType::L2Unexpanded;
  } else if (metric_string == "L2SqrtUnexpanded") {
    result = raft::distance::DistanceType::L2SqrtUnexpanded;
  } else if (metric_string == "InnerProduct") {
    result = raft::distance::DistanceType::InnerProduct;
  } else if (metric_string == "Linf") {
    result = raft::distance::DistanceType::Linf;
  } else if (metric_string == "Canberra") {
    result = raft::distance::DistanceType::Canberra;
  } else if (metric_string == "LpUnexpanded") {
    result = raft::distance::DistanceType::LpUnexpanded;
  } else if (metric_string == "CorrelationExpanded") {
    result = raft::distance::DistanceType::CorrelationExpanded;
  } else if (metric_string == "JaccardExpanded") {
    result = raft::distance::DistanceType::JaccardExpanded;
  } else if (metric_string == "HeillingerExpanded") {
    result = raft::distance::DistanceType::HeillingerExpanded;
  } else if (metric_string == "Haversine") {
    result = raft::distance::DistanceType::Haversine;
  } else if (metric_string == "BrayCurtis") {
    result = raft::distance::DistanceType::BrayCurtis;
  } else if (metric_string == "JensenShannon") {
    result = raft::distance::DistanceType::JensenShannon;
  } else if (metric_string == "HammingUnexpanded") {
    result = raft::distance::DistanceType::HammingUnexpanded;
  } else if (metric_string == "KLDivergence") {
    result = raft::distance::DistanceType::KLDivergence;
  } else if (metric_string == "RusselRaoExpanded") {
    result = raft::distance::DistanceType::RusselRaoExpanded;
  } else if (metric_string == "DiceExpanded") {
    result = raft::distance::DistanceType::DiceExpanded;
  } else if (metric_string == "Precomputed") {
    result = raft::distance::DistanceType::Precomputed;
  } else {
    RAFT_FAIL("Unrecognized metric type %s", metric_string.c_str());
  }
  return result;
}

[[nodiscard]] inline auto codebook_string_to_raft_codebook_gen(std::string const& codebook_string) {
  auto result = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
  if (codebook_string == "PER_SUBSPACE") {
    result = raft::neighbors::ivf_pq::codebook_gen::PER_SUBSPACE;
  } else if (codebook_string == "PER_CLUSTER") {
    result = raft::neighbors::ivf_pq::codebook_gen::PER_CLUSTER;
  } else {
    RAFT_FAIL("Unrecognized codebook type %s", codebook_string.c_str());
  }
  return result;
}
[[nodiscard]] inline auto build_algo_string_to_cagra_build_algo(std::string
    const& algo_string) {
  auto result = raft::neighbors::cagra::graph_build_algo::IVF_PQ;
  if (algo_string == "IVF_PQ") {
    result = raft::neighbors::cagra::graph_build_algo::IVF_PQ;
  } else if (algo_string == "NN_DESCENT") {
    result = raft::neighbors::cagra::graph_build_algo::NN_DESCENT;
  } else {
    RAFT_FAIL("Unrecognized CAGRA build algo %s", algo_string.c_str());
  }
  return result;
}

[[nodiscard]] inline auto search_algo_string_to_cagra_search_algo(std::string
    const& algo_string) {
  auto result = raft::neighbors::cagra::search_algo::AUTO;
  if (algo_string == "SINGLE_CTA") {
    result = raft::neighbors::cagra::search_algo::SINGLE_CTA;
  } else if (algo_string == "MULTI_CTA") {
    result = raft::neighbors::cagra::search_algo::MULTI_CTA;
  } else if (algo_string == "MULTI_KERNEL") {
    result = raft::neighbors::cagra::search_algo::MULTI_KERNEL;
  } else if (algo_string == "AUTO") {
    result = raft::neighbors::cagra::search_algo::AUTO;
  } else {
    RAFT_FAIL("Unrecognized CAGRA search algo %s", algo_string.c_str());
  }
  return result;
}

[[nodiscard]] inline auto hashmap_mode_string_to_cagra_hashmap_mode(std::string
    const& mode_string) {
  auto result = raft::neighbors::cagra::hash_mode::AUTO;
  if (mode_string == "HASH") {
    result = raft::neighbors::cagra::hash_mode::HASH;
  } else if (mode_string == "SMALL") {
    result = raft::neighbors::cagra::hash_mode::SMALL;
  } else if (mode_string == "AUTO") {
    result = raft::neighbors::cagra::hash_mode::AUTO;
  } else {
    RAFT_FAIL("Unrecognized CAGRA hash mode %s", mode_string.c_str());
  }
  return result;
}

[[nodiscard]] inline auto dtype_string_to_cuda_dtype(std::string
    const& dtype_string) {
  auto result = CUDA_R_32F;
  if (dtype_string == "CUDA_R_16F") {
    result = CUDA_R_16F;
  } else if (dtype_string == "CUDA_C_16F") {
    result = CUDA_C_16F;
  } else if (dtype_string == "CUDA_R_16BF") {
    result = CUDA_R_16BF;
  } else if (dtype_string == "CUDA_R_32F") {
    result = CUDA_R_32F;
  } else if (dtype_string == "CUDA_C_32F") {
    result = CUDA_C_32F;
  } else if (dtype_string == "CUDA_R_64F") {
    result = CUDA_R_64F;
  } else if (dtype_string == "CUDA_C_64F") {
    result = CUDA_C_64F;
  } else if (dtype_string == "CUDA_R_8I") {
    result = CUDA_R_8I;
  } else if (dtype_string == "CUDA_C_8I") {
    result = CUDA_C_8I;
  } else if (dtype_string == "CUDA_R_8U") {
    result = CUDA_R_8U;
  } else if (dtype_string == "CUDA_C_8U") {
    result = CUDA_C_8U;
  } else if (dtype_string == "CUDA_R_32I") {
    result = CUDA_R_32I;
  } else if (dtype_string == "CUDA_C_32I") {
    result = CUDA_C_32I;
  } else if (dtype_string == "CUDA_R_8F_E4M3") {
    result = CUDA_R_8F_E4M3;
  } else if (dtype_string == "CUDA_R_8F_E5M2") {
    result = CUDA_R_8F_E5M2;
  } else {
    RAFT_FAIL("Unrecognized dtype %s", dtype_string.c_str());
  }
  return result;
}

// Given a generic config without RAFT symbols, convert to RAFT index build
// parameters
template <raft_index_kind IndexKind>
[[nodiscard]] auto config_to_index_params(raft_knowhere_config const& raw_config) {
  RAFT_EXPECTS(raw_config.index_type == IndexKind, "Incorrect index type for this index");
  auto config = validate_config(raw_config);
  auto result = raft_index_params_t<IndexKind>{};

  result.metric = metric_string_to_raft_distance_type(config.metric);
  result.metric_arg = config.metric_arg;
  result.add_data_on_build = config.add_data_on_build;

  if constexpr (IndexKind == raft_index_kind::ivf_flat || IndexKind ==
      raft_index_kind::ivf_pq) {
    result.n_lists = *(config.nlist);
    result.kmeans_n_iters = *(config.kmeans_n_iters);
    result.kmeans_trainset_fraction = *(config.kmeans_trainset_fraction);
    result.conservative_memory_allocation = *(config.conservative_memory_allocation);
  }
  if constexpr (IndexKind == raft_index_kind::ivf_flat) {
    result.adaptive_centers = *(config.adaptive_centers);
  }
  if constexpr (IndexKind == raft_index_kind::ivf_pq) {
    result.pq_dim = *(config.m);
    result.pq_bits = *(config.nbits);
    result.codebook_kind = codebook_string_to_raft_codebook_gen(config.codebook_gen);
    result.force_random_rotation = *(config.force_random_rotation);
  }
  if constexpr (IndexKind == raft_index_kind::cagra) {
    result.intermediate_graph_degree = *(config.intermediate_graph_degree);
    result.graph_degree = *(config.graph_degree);
    result.build_algo = build_algo_string_to_cagra_build_algo(*(config.build_algo));
    result.nn_descent_niter = *(config.nn_descent_niter);
  }
  return result;
}

// Given a generic config without RAFT symbols, convert to RAFT index search
// parameters
template <raft_index_kind IndexKind>
[[nodiscard]] auto config_to_search_params(raft_knowhere_config const& raw_config) {
  RAFT_EXPECTS(raw_config.index_type == IndexKind, "Incorrect index type for this index");
  auto config = validate_config(raw_config);
  auto result = raft_search_params_t<IndexKind>{};
  if constexpr (IndexKind == raft_index_kind::ivf_flat || IndexKind == raft_index_kind::ivf_pq) {
    result.n_probes = *(config.nprobe);
  }
  if constexpr (IndexKind == raft_index_kind::ivf_pq) {
    result.lut_dtype = dtype_string_to_cuda_dtype(*(config.lookup_dtype));
    result.internal_distance_dtype = dtype_string_to_cuda_dtype(*(config.internal_distance_dtype));
    result.preferred_shmem_carveout = *(config.preferred_shmem_carveout);
  }
  if constexpr (IndexKind == raft_index_kind::cagra) {
    result.max_queries = *(config.max_queries);
    result.max_iterations = *(config.max_iterations);
    result.search_algo = search_algo_string_to_cagra_search_algo(*(config.search_algo));
    result.team_size = *(config.team_size);
    result.search_width = *(config.search_width);
    result.min_iterations = *(config.min_iterations);
    result.thread_block_size = *(config.thread_block_size);
    result.hashmap_mode = hashmap_mode_string_to_cagra_hashmap_mode(*(config.hashmap_mode));
    result.hashmap_min_bitlen = *(config.hashmap_min_bitlen);
    result.hashmap_max_fill_rate = *(config.hashmap_max_fill_rate);
  }
  return result;
}

// This struct is used to connect knowhere to a RAFT index. The implementation
// is provided here, but this header should never be directly included in
// another knowhere header. This ensures that RAFT symbols are not exposed in
// any knowhere header.
template <raft_index_kind IndexKind>
struct raft_knowhere_index {
  auto static constexpr index_kind = IndexKind;

  using raft_index_type = raft_index_t<index_kind>;
  using index_params_type = raft_index_params_t<index_kind>;
  using search_params_type = raft_search_params_t<index_kind>;
  using data_type = raft_data_t<index_kind>;
  auto is_trained() const {
    return index_.has_value();
  }

  auto train(raft_knowhere_config const& config, data_type const* data, std::int64_t row_count, std::int64_t feature_count) {
    auto index_params = config_to_index_params<index_kind>(config);
    auto const& res = raft::device_resources_manager::get_device_resources();
    auto host_data = raft::make_host_matrix_view(data, row_count, feature_count);
    auto device_data_storage = raft::make_device_matrix(res, row_count, feature_count);
    auto device_data = device_data_storage.view();
    raft::copy(res, device_data, host_data);
    index_ = raft_index_type::build(res, index_params, device_data);
  }

  auto add(data_type const* data, std::int64_t row_count, std::int64_t
      feature_count, std::int64_t* new_ids=nullptr) {
    auto const& res = raft::device_resources_manager::get_device_resources();
    auto host_data = raft::make_host_matrix_view(data, row_count, feature_count);
    auto device_data_storage = raft::make_device_matrix(res, row_count, feature_count);
    auto device_data = device_data_storage.view();
    raft::copy(res, device_data, host_data);
    // TODO(wphicks): handle new_ids
    if (index_) {
      raft_index_type::extend(res, *index_, device_data);
    } else {
      // TODO(wphicks): throw exception
    }
  }

  auto search() const {}
  auto search(raft_knowhere_config const& config, data_type const* data, std::int64_t row_count, std::int64_t feature_count) {
  }
  auto range_search() const {}
  auto get_vector_by_id() const {}
  auto serialize() const {}
  auto deserialize() {}
  auto deserialize_from_file() {}
  auto dim() const {}
  auto size() const {}
 private:
  // TODO(wphicks): Put inside streamsafe_wrapper
  std::optional<raft_index_type> index_;
};

} // namespace raft_knowhere

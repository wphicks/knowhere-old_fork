#pragma once
#include <cstdint>
#include <type_traits>
#include <raft/core/device_resources_manager.hpp>
#include "index/raft/raft_index_kind.hpp"
#include "index/raft/raft_index.cuh"

namespace detail {

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

}

template <raft_index_kind IndexKind>
using raft_index_t = typename detail::raft_index_type_mapper<true, IndexKind>::type;

template <raft_index_kind IndexKind>
using raft_data_t = typename detail::raft_index_type_mapper<true, IndexKind>::data_t;

template <raft_index_kind IndexKind>
using raft_index_params_t = typename detail::raft_index_type_mapper<true, IndexKind>::index_params_type;
template <raft_index_kind IndexKind>
using raft_search_params_t = typename detail::raft_index_type_mapper<true, IndexKind>::search_params_type;

template <raft_index_kind IndexKind>
struct raft_knowhere_index {
  auto static constexpr index_kind = IndexKind;

  using raft_index_type = raft_index_t<index_kind>;
  using index_params_type = raft_index_params_t<index_kind>;
  using search_params_type = raft_search_params_t<index_kind>;
  using data_type = raft_data_t<index_kind>;
  auto is_trained() {
    return index_.has_value();
  }

  auto train(index_params_type const& index_params, data_type const* data, std::int64_t row_count, std::int64_t feature_count) {
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

  auto search() {}
  auto range_search() {}
  auto get_vector_by_id() {}
  auto serialize() {}
  auto deserialize() {}
  auto deserialize_from_file() {}
  auto dim() {}
  auto size() {}
 private:
  // TODO(wphicks): Put inside streamsafe_wrapper
  std::optional<raft_index_type> index_;
};

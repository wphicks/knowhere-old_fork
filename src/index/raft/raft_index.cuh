#include <cstddef>
#include <cstdint>
#include <optional>
#include <type_traits>
#include "index/raft/raft_index_kind.hpp"

template <template <typename...> typename underly_index, typename... raft_index_args>
struct raft_index {
  using vector_index_type = underlying_index<raft_index_args...>;
  auto static constexpr vector_index_kind = []() {
    if constexpr (std::is_same_v<
      vector_index_type,
      raft::neighbors::ivf_flat::index<raft_index_args...>
    >) {
      return raft_index_kind::ivf_flat;
    } else if constexpr (std::is_same_v<
      vector_index_type,
      raft::neighbors::ivf_pq::index<raft_index_args...>
    >) {
      return raft_index_kind::ivf_pq;
    } else if constexpr (std::is_same_v<
      vector_index_type,
      raft::neighbors::cagra::index<raft_index_args...>
    >) {
      return raft_index_kind::cagra;
    } else {
      static_assert(std::is_same_v<
        vector_index_type,
        raft::neighbors::ivf_flat::index<raft_index_args...>
      >, "Unsupported index template passed to raft_index");
    }
  }();

  using index_params_type = std::conditional_t<
    vector_index_kind == raft_index_kind::ivf_flat,
    raft::neighbors::ivf_flat::index_params,
    std::conditional_t<
      vector_index_kind == raft_index_kind::ivf_pq,
      raft::neighbors::ivf_pq::index_params,
      std::conditional_t<
        vector_index_kind == raft_index_kind::cagra,
        raft::neighbors::cagra::index_params,
        // Should never get here; precluded by static assertion above
        raft::neighbors::ivf_flat::index_params
      >
    >
  >;
  using search_params_type = std::conditional_t<
    vector_index_kind == raft_index_kind::ivf_flat,
    raft::neighbors::ivf_flat::search_params,
    std::conditional_t<
      vector_index_kind == raft_index_kind::ivf_pq,
      raft::neighbors::ivf_pq::search_params,
      std::conditional_t<
        vector_index_kind == raft_index_kind::cagra,
        raft::neighbors::cagra::search_params,
        // Should never get here; precluded by static assertion above
        raft::neighbors::ivf_flat::search_params
      >
    >
  >;
 private:
  using self_type = raft_index<underlying_index, raft_index_args...>;

 public:

  auto& get_vector_index() { return vector_index_; }
  auto const& get_vector_index() const { return vector_index_; }

  // TODO(wphicks): Set up tiered index

  template <typename T, typename IdxT>
  auto static build(
    raft::device_resources const& res,
    index_params_type const& index_params,
    raft::device_matrix_view<T const, IdxT> data
  ) {
    if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
      return self_type{
        raft::neighbors::ivf_flat::build<T, IdxT>(
          res,
          index_params,
          data
        )
      };
    } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
      return self_type{
        raft::neighbors::ivf_pq::build<T, IdxT>(
          res,
          index_params,
          data
        )
      };
    } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
      return self_type{
        raft::neighbors::cagra::build<T>(
          res,
          index_params,
          data
        )
      };
    }
  }

  template <typename T, typename IdxT, typename FilterT=std::nullptr_t>
  auto static search(
    raft::device_resources const& res,
    self_type const& index,
    search_params_type const& search_params,
    raft::device_matrix_view<T const, IdxT> queries,
    raft::device_matrix_view<IdxT, IdxT> neighbors,
    raft::device_matrix_view<float, IdxT> distances,
    float refine_ratio = 1.0f,
    IdxT k_offset=IdxT{},
    std::optional<raft::device_matrix_view<const T, IdxT>> dataset = std::nullopt,
    FilterT filter = nullptr
  ) {
    auto const& underlying_index = index.get_vector_index();

    auto k = neighbors.extent(1);
    auto k_tmp = k + k_offset;
    if (refine_ratio > 1.0f) {
      k_tmp *= refine_ratio;
    }

    auto neighbors_tmp = neighbors;
    auto distances_tmp = distances;
    auto neighbors_storage = std::optional<raft::device_matrix<IdxT, IdxT>>{};
    auto distances_storage = std::optional<raft::device_matrix<float, IdxT>>{};

    if (k_tmp > k) {
      neighbors_storage = raft::make_device_matrix<IdxT, IdxT>(
        res,
        queries.extent(0),
        k_tmp
      );
      neighbors_tmp = neighbors_storage.view();
      distances_storage = raft::make_device_matrix<float, IdxT>(
        res,
        queries.extent(0),
        k_tmp
      );
      distances_tmp = distances_storage.view();
    }

    if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
      if constexpr (std::is_same_v(FilterT, std::nullptr_t)){
        raft::neighbors::ivf_flat::search<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp
        );
      } else {
        raft::neighbors::ivf_flat::search_with_filtering<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp,
          filter
        );
      }
    } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
      if constexpr (std::is_same_v(FilterT, std::nullptr_t)){
        raft::neighbors::ivf_pq::search_<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp
        );
      } else {
        raft::neighbors::ivf_pq::search_with_filtering<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp,
          filter
        );
      }
    } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
      if constexpr (std::is_same_v(FilterT, std::nullptr_t)){
        raft::neighbors::cagra::search<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp
        );
      } else {
        raft::neighbors::cagra::search_with_filtering<T, IdxT>(
          res,
          search_params,
          underlying_index,
          queries,
          neighbors_tmp,
          distances_tmp,
          filter
        );
      }
    }
    if (refine_ratio > 1.0f) {
      if (dataset.has_value()) {
        // TODO (wphicks): Check if this is in-place
        raft::neighbors::refine(
          res,
          *dataset,
          queries,
          neighbors_tmp,
          neighbors,
          distances,
          underlying_index.metric()
        );
      } else {
        RAFT_WARN(
          "Refinement requested, but no dataset provided. "
          "Ignoring refinement request."
        );
      }
    }
    if (k_tmp > k) {
      // TODO(wphicks): Take into account k_offset
      raft::copy(
        res,
        neighbors,
        raft::make_device_matrix_view(neighbors_tmp.data(), neighbors.extent(0), k)
      );
      raft::copy(
        res,
        distances,
        raft::make_device_matrix_view(distances_tmp.data(), distances.extent(0), k)
      );
    }
  }

  template <typename T, typename IdxT>
  auto static extend(
    raft::device_resources const& res,
    self_type const& index,
    raft::device_matrix_view<T const, IdxT> new_vectors,
    std::optional<raft::device_matrix_view<IdxT const, IdxT>> new_ids
  ) {
    auto const& underlying_index = index.get_vector_index();

    if constexpr (vector_index_kind == raft_index_kind::ivf_flat) {
      return raft::neighbors::ivf_flat::extend<T, IdxT>(
        res,
        underlying_index,
        new_vectors,
        new_ids,
        underlying_index
      );
    } else if constexpr (vector_index_kind == raft_index_kind::ivf_pq) {
      return raft::neighbors::ivf_pq::extend<T, IdxT>(
        res,
        underlying_index,
        new_vectors,
        new_ids,
        underlying_index
      );
    } else if constexpr (vector_index_kind == raft_index_kind::cagra) {
      return raft::neighbors::cagra::extend<T, IdxT>(
        res,
        underlying_index,
        new_vectors,
        new_ids,
        underlying_index
      );
    }
  }

 private:
  vector_index_type vector_index_;

  explicit raft_index(vector_index_type&& vector_index)
    : vector_index_{std::move(vector_index)} {}
};

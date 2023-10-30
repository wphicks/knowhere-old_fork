#ifndef GPU_RAFT_H
#define GPU_RAFT_H

#include <cstdint>
#include <exception>
#include <numeric>
#include <tuple>
#include <vector>

#include "index/gpu_raft/gpu_raft_cagra_config.h"
#include "index/gpu_raft/gpu_raft_ivf_flat_config.h"
#include "index/gpu_raft/gpu_raft_ivf_pq_config.h"

#include "common/raft/proto/raft_index_kind.hpp"
#include "common/raft/integration/raft_knowhere_index.hpp"
#include "common/raft/integration/raft_knowhere_config.hpp"

#include "knowhere/comp/index_param.h"
#include "knowhere/expected.h"
#include "knowhere/factory.h"
#include "knowhere/log.h"
#include "knowhere/utils.h"

namespace knowhere {

auto static constexpr cuda_concurrent_size = std::uint32_t{32};

template <raft_proto::raft_index_kind K>
struct KnowhereConfigType {};

template <>
struct KnowhereConfigType<raft_proto::raft_index_kind::ivf_flat> {
  using Type = GpuRaftIvfFlatConfig;
};

template <>
struct KnowhereConfigType<raft_proto::raft_index_kind::ivf_pq> {
  using Type = GpuRaftIvfPqConfig;
};

template <>
struct KnowhereConfigType<raft_proto::raft_index_kind::cagra> {
  using Type = GpuRaftCagraConfig;
};

template <raft_proto::raft_index_kind K>
struct GpuRaftIndexNode : public IndexNode {
  auto static constexpr index_kind = K;
  using knowhere_config_type = typename KnowhereConfigType<index_kind>::Type;

  GpuRaftIndexNode(int32_t, const Object& object) : index_{} {}

  Status
  Train(const DataSet& dataset, const Config& cfg) override {
    auto result = Status::success;
    auto raft_cfg = raft_knowhere::raft_knowhere_config{};
    try {
      raft_cfg = to_raft_knowhere_config(
        static_cast<const knowhere_config_type&>(cfg)
      );
    } catch (const std::exception& e) {
      LOG_KNOWHERE_ERROR_ << e.what();
      result = Status::invalid_args;
    }
    if (index_.is_trained()) {
      result = Status::index_already_trained;
    }
    if (result == Status::success) {
      auto rows = dataset.GetRows();
      auto dim = dataset.GetDim();
      auto const* data = reinterpret_cast<float const*>(dataset.GetTensor());
      try {
        index_.train(raft_cfg, data, rows, dim);
        index_.synchronize();
      } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << e.what();
        result = Status::raft_inner_error;
      }
    }
    return result;
  }

  Status
  Add(const DataSet& dataset, const Config& cfg) override {
    if constexpr(index_kind == raft_proto::raft_index_kind::cagra || index_kind == raft_proto::raft_index_kind::ivf_pq) {
      return Status::success;
    } else {
      auto rows = dataset.GetRows();
      auto dim = dataset.GetDim();
      auto const* data = reinterpret_cast<float const*>(dataset.GetTensor());
      auto new_ids = std::vector<int64_t>(rows);
      std::iota(std::begin(new_ids), std::end(new_ids), index_.size());
      try {
        index_.add(data, rows, dim, new_ids.data());
        index_.synchronize();
      } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << e.what();
        return Status::raft_inner_error;
      }
      return Status::success;
    }
  }

  expected<DataSetPtr>
  Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
    auto result = Status::success;
    auto raft_cfg = raft_knowhere::raft_knowhere_config{};
    auto err_msg = std::string{};
    try {
      raft_cfg = to_raft_knowhere_config(
        static_cast<const knowhere_config_type&>(cfg)
      );
    } catch (const std::exception& e) {
      err_msg = std::string{e.what()};
      LOG_KNOWHERE_ERROR_ << e.what();
      result = Status::invalid_args;
    }
    if (result == Status::success) {
      try {
        auto rows = dataset.GetRows();
        auto dim = dataset.GetDim();
        auto const* data = reinterpret_cast<float const*>(dataset.GetTensor());
        auto search_result = index_.search(
          raft_cfg,
          data,
          rows,
          dim,
          bitset.data(),
          bitset.byte_size(),
          bitset.size()
        );
        index_.synchronize();
        return GenResultDataSet(
          rows,
          raft_cfg.k,
          std::get<0>(search_result),
          std::get<1>(search_result)
        );
      } catch (const std::exception& e) {
        err_msg = std::string{e.what()};
        LOG_KNOWHERE_ERROR_ << e.what();
        result = Status::raft_inner_error;
      }
    }
    return expected<DataSetPtr>::Err(result, err_msg.c_str());
  }

  expected<DataSetPtr>
  RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
      return expected<DataSetPtr>::Err(Status::not_implemented, "RangeSearch not implemented");
  }

  expected<DataSetPtr>
  GetVectorByIds(const DataSet& dataset) const override {
    return expected<DataSetPtr>::Err(Status::not_implemented, "GetVectorByIds not implemented");
  }

  bool
  HasRawData(const std::string& metric_type) const override {
    if constexpr (index_kind == raft_proto::raft_index_kind::ivf_flat) {
      return !IsMetricType(metric_type, metric::COSINE);
    } else {
      return false;
    }
  }

  Status
  Serialize(BinarySet& binset) const override {
    auto result = Status::success;
    std::stringbuf buf;
    if (!index_.is_trained()) {
      result = Status::empty_index;
    } else {
      std::ostream os(&buf);

      try {
        index_.serialize(os);
        index_.synchronize();
      } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << e.what();
        result = Status::raft_inner_error;
      }
      os.flush();
    }
    if (result == Status::success) {
      std::shared_ptr<uint8_t[]> index_binary(new (std::nothrow) uint8_t[buf.str().size()]);
      memcpy(index_binary.get(), buf.str().c_str(), buf.str().size());
      binset.Append(this->Type(), index_binary, buf.str().size());
    }
    return result;
  }

  Status
  Deserialize(const BinarySet& binset, const Config& config) override {
    auto result = Status::success;
    std::stringbuf buf;
    auto binary = binset.GetByName(this->Type());
    if (binary == nullptr) {
        LOG_KNOWHERE_ERROR_ << "Invalid binary set.";
        result = Status::invalid_binary_set;
    } else {
      buf.sputn((char*)binary->data.get(), binary->size);
      std::istream is(&buf);

      try {
        index_ = raft_knowhere_index_type::deserialize(is);
        index_.synchronize();
      } catch (const std::exception& e) {
        LOG_KNOWHERE_ERROR_ << e.what();
        result = Status::raft_inner_error;
      }
      is.sync();
    }
    return result;
  }

  Status
  DeserializeFromFile(const std::string& filename, const Config& config) {
    // TODO(wphicks): This is simple to implement by just opening an ifstream
    // and casting to an istream then following the same steps as above
    LOG_KNOWHERE_ERROR_ << "RaftIvfIndex doesn't support Deserialization from file.";
    return Status::not_implemented;
  }


  std::unique_ptr<BaseConfig>
  CreateConfig() const override {
    return std::make_unique<knowhere_config_type>();
  }


  expected<DataSetPtr>
  GetIndexMeta(const Config& cfg) const override {
    return expected<DataSetPtr>::Err(Status::not_implemented, "GetIndexMeta not implemented");
  }

  int64_t
  Dim() const override {
    return index_.dim();
  }

  int64_t
  Size() const override {
    return 0;
  }

  int64_t
  Count() const override {
    return index_.size();
  }

  std::string
  Type() const override {
      if constexpr (index_kind == raft_proto::raft_index_kind::ivf_flat) {
          return knowhere::IndexEnum::INDEX_RAFT_IVFFLAT;
      } else if constexpr (index_kind == raft_proto::raft_index_kind::ivf_pq) {
          return knowhere::IndexEnum::INDEX_RAFT_IVFPQ;
      } else if constexpr (index_kind == raft_proto::raft_index_kind::cagra) {
          return knowhere::IndexEnum::INDEX_RAFT_CAGRA;
      }
  }


 private:
  using raft_knowhere_index_type = typename raft_knowhere::raft_knowhere_index<K>;

  raft_knowhere_index_type index_;
};

extern template struct GpuRaftIndexNode<raft_proto::raft_index_kind::ivf_flat>;
extern template struct GpuRaftIndexNode<raft_proto::raft_index_kind::ivf_pq>;
extern template struct GpuRaftIndexNode<raft_proto::raft_index_kind::cagra>;

using GpuRaftIvfFlatIndexNode = GpuRaftIndexNode<raft_proto::raft_index_kind::ivf_flat>;
using GpuRaftIvfPqIndexNode = GpuRaftIndexNode<raft_proto::raft_index_kind::ivf_pq>;
using GpuRaftCagraIndexNode = GpuRaftIndexNode<raft_proto::raft_index_kind::cagra>;

}  // namespace knowhere
   //
#endif /* GPU_RAFT_H */

#include <omp.h>

#include "hnswlib/hnswalg.h"
#include "hnswlib/hnswlib.h"
#include "index/hnsw/hnsw_config.h"
#include "knowhere/knowhere.h"
namespace knowhere {
class HnswIndexNode : public IndexNode {
 public:
    HnswIndexNode() : index_(nullptr) {
    }
    virtual Error
    Build(const DataSet& dataset, const Config& cfg) override {
        auto res = Train(dataset, cfg);
        if (res != Error::success)
            return res;
        res = Add(dataset, cfg);
        return res;
    }
    virtual Error
    Train(const DataSet& dataset, const Config& cfg) override {
        if (index_)
            delete index_;
        auto rows = dataset.GetRows();
        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        hnswlib::SpaceInterface<float>* space = NULL;
        if (hnsw_cfg.metric_type == "L2") {
            space = new (std::nothrow) hnswlib::L2Space(hnsw_cfg.dim);
        }
        if (hnsw_cfg.metric_type == "IP") {
            space = new (std::nothrow) hnswlib::InnerProductSpace(hnsw_cfg.dim);
        }
        if (space == NULL)
            return Error::invalid_metric_type;

        index_ = new (std::nothrow) hnswlib::HierarchicalNSW<float>(space, rows, hnsw_cfg.M, hnsw_cfg.efConstruction);
        return Error::success;
    }
    virtual Error
    Add(const DataSet& dataset, const Config& cfg) override {
        if (!index_) {
            return Error::empty_index;
        }

        auto rows = dataset.GetRows();
        auto tensor = dataset.GetTensor();
        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        index_->addPoint(tensor, 0);
        if (hnsw_cfg.build_thread_num > 0)
            omp_set_num_threads(hnsw_cfg.build_thread_num);
#pragma omp parallel for
        for (int i = 1; i < rows; ++i) {
            index_->addPoint((static_cast<const float*>(tensor) + hnsw_cfg.dim * i), i);
        }
        return Error::success;
    }
    virtual expected<DataSetPtr, Error>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
        if (!index_) {
            return unexpected(Error::empty_index);
        }

        auto rows = dataset.GetRows();
        auto tensor = dataset.GetTensor();
        auto hnsw_cfg = static_cast<const HnswConfig&>(cfg);
        auto dim = hnsw_cfg.dim;
        auto k = hnsw_cfg.k;
        auto p_id = new int64_t[k * rows];
        auto p_dist = new float[k * rows];

        hnswlib::SearchParam param{(size_t)hnsw_cfg.ef};
        bool transform = (index_->metric_type_ == 1);  // InnerProduct: 1
        if (hnsw_cfg.query_thread_num > 0)
            omp_set_num_threads(hnsw_cfg.query_thread_num);
#pragma omp parallel for
        for (unsigned int i = 0; i < rows; ++i) {
            auto single_query = (float*)tensor + i * dim;
            std::priority_queue<std::pair<float, hnswlib::labeltype>> rst;
            auto dummy_stat = hnswlib::StatisticsInfo();
            rst = index_->searchKnn(single_query, k, bitset, dummy_stat, &param);
            size_t rst_size = rst.size();

            auto p_single_dis = p_dist + i * k;
            auto p_single_id = p_id + i * k;
            int idx = rst_size - 1;
            while (!rst.empty()) {
                auto& it = rst.top();
                p_single_dis[idx] = transform ? (1 - it.first) : it.first;
                p_single_id[idx] = it.second;
                rst.pop();
                idx--;
            }

            for (idx = rst_size; idx < k; idx++) {
                p_single_dis[idx] = float(1.0 / 0.0);
                p_single_id[idx] = -1;
            }
        }

        auto res = std::make_shared<DataSet>();
        res->SetIds(p_id);
        res->SetDistance(p_dist);
        return res;
    }
    virtual expected<DataSetPtr, Error>
    GetVectorByIds(const DataSet& dataset, const Config& cfg) const override {
        if (!index_) {
            return unexpected(Error::empty_index);
        }

        auto dim = dataset.GetDim();
        auto rows = dataset.GetRows();
        auto ids = dataset.GetIds();

        float* p_x = nullptr;
        try {
            p_x = new float[dim * rows];
            for (int64_t i = 0; i < rows; i++) {
                int64_t id = ids[i];
                assert(id >= 0 && id < (int64_t)index_->cur_element_count);
                memcpy(p_x + i * dim, index_->getDataByInternalId(id), dim * sizeof(float));
            }
        } catch (...) {
            std::unique_ptr<float> auto_delete_px(p_x);
        }
        auto res = std::make_shared<DataSet>();
        res->SetTensor(p_x);
        return res;
    }
    virtual Error
    Serialization(BinarySet& binset) const override {
        if (!index_) {
            return Error::empty_index;
        }

        try {
            MemoryIOWriter writer;
            index_->saveIndex(writer);
            std::shared_ptr<uint8_t[]> data(writer.data_);

            binset.Append("HNSW", data, writer.rp);

        } catch (...) {
            return Error::hnsw_inner_error;
        }

        return Error::success;
    }
    virtual Error
    Deserialization(const BinarySet& binset) override {
        if (index_)
            delete index_;
        try {
            auto binary = binset.GetByName("HNSW");

            MemoryIOReader reader;
            reader.total = binary->size;
            reader.data_ = binary->data.get();

            hnswlib::SpaceInterface<float>* space = nullptr;
            index_ = new (std::nothrow) hnswlib::HierarchicalNSW<float>(space);
            index_->loadIndex(reader);
        } catch (...) {
            return Error::hnsw_inner_error;
        }

        return Error::success;
    }

    virtual std::unique_ptr<BaseConfig>
    CreateConfig() const override {
        return std::make_unique<HnswConfig>();
    }
    virtual int64_t
    Dims() const override {
        if (!index_)
            return (*static_cast<size_t*>(index_->dist_func_param_));
        return 0;
    }
    virtual int64_t
    Size() const override {
        if (!index_)
            return 0;
        return index_->cal_size();
    }
    virtual int64_t
    Count() const override {
        if (!index_)
            return 0;
        return index_->cur_element_count;
    }
    virtual std::string
    Type() const override {
        return "HNSW";
    }
    virtual ~HnswIndexNode() {
        if (index_)
            delete index_;
    }

 private:
    hnswlib::HierarchicalNSW<float>* index_;
};

KNOWHERE_REGISTER_GLOBAL(HNSW, []() { return Index<HnswIndexNode>::Create(); });

}  // namespace knowhere

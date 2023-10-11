/**
 * SPDX-FileCopyrightText: Copyright (c) 2023,NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef IVF_RAFT_CUH
#define IVF_RAFT_CUH


namespace knowhere {

namespace detail {
}  // namespace detail

template <typename T>
struct KnowhereConfigType {};

template <>
struct KnowhereConfigType<detail::raft_ivf_flat_index> {
    typedef RaftIvfFlatConfig Type;
};
template <>
struct KnowhereConfigType<detail::raft_ivf_pq_index> {
    typedef RaftIvfPqConfig Type;
};

template <typename T>
class RaftIvfIndexNode : public IndexNode {
 public:
    RaftIvfIndexNode(const int32_t& /*version*/, const Object& object) : device_id_{-1}, gpu_index_{} {
    }

    Status
    Train(const DataSet& dataset, const Config& cfg) override {
    }

    Status
    Add(const DataSet& dataset, const Config& cfg) override {
    }

    expected<DataSetPtr>
    Search(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
    }

    expected<DataSetPtr>
    RangeSearch(const DataSet& dataset, const Config& cfg, const BitsetView& bitset) const override {
    }

    expected<DataSetPtr>
    GetVectorByIds(const DataSet& dataset) const override {
    }

    bool
    HasRawData(const std::string& metric_type) const override {
    }

    expected<DataSetPtr>
    GetIndexMeta(const Config& cfg) const override {
    }

    Status
    Serialize(BinarySet& binset) const override {
    }

    Status
    Deserialize(const BinarySet& binset, const Config& config) override {
    }

    Status
    DeserializeFromFile(const std::string& filename, const Config& config) {
        return Status::not_implemented;
    }

    std::unique_ptr<BaseConfig>
    CreateConfig() const override {
    }

    int64_t
    Dim() const override {
    }

    int64_t
    Size() const override {
    }

    int64_t
    Count() const override {
    }

    std::string
    Type() const override {
    }
};

}  // namespace knowhere
#endif /* IVF_RAFT_CUH */


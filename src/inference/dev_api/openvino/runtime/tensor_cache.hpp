// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <atomic>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "blocked_tensor_desc.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {
// namespace one_plugin {

std::string OPENVINO_RUNTIME_API get_weights_id(const std::shared_ptr<ov::op::v0::Constant>& constant);

class OPENVINO_RUNTIME_API WeightsCache {
public:
    typedef std::shared_ptr<WeightsCache> Ptr;

    struct TensorInfo {
        typedef std::shared_ptr<TensorInfo> Ptr;

        TensorInfo(ov::Tensor tensor, BlockedTensorDesc::CPtr tensor_desc) : m_tensor(std::move(tensor)), m_tensor_desc(std::move(tensor_desc)) {}

        ov::Tensor m_tensor;
        BlockedTensorDesc::CPtr m_tensor_desc;
    };

    std::multimap<std::string, TensorInfo>::const_iterator get(std::string id) const;
    std::multimap<std::string, TensorInfo>::const_iterator findOrCreate(std::string id, std::function<TensorInfo::Ptr(void)> create);
    std::multimap<std::string, WeightsCache::TensorInfo>::const_iterator replaceWith(
        std::multimap<std::string, WeightsCache::TensorInfo>::const_iterator pos, WeightsCache::TensorInfo::Ptr item);

    // TODO: make private
    std::multimap<std::string, TensorInfo> m_storage;
private:
};

// class WeightsCache {
//     using ITensorPtr = std::shared_ptr<ITensor>;

//     struct TensorInfo {
//         typedef std::shared_ptr<TensorInfo> Ptr;

//         TensorInfo(ITensorPtr tensor, bool valid) : shared_tensor(tensor), valid(valid) {}

//         // std::mutex guard;
//         // std::atomic<bool> valid;
//         std::shared_ptr<ITensor> m_tensor;
//         std::shared_ptr<ITensor> m_tensor;
//     };

// public:
//     typedef std::shared_ptr<WeightsCache> Ptr;

//     class SharedTensor {
//     public:
//         typedef std::shared_ptr<SharedTensor> Ptr;

//         SharedTensor(std::unique_lock<std::mutex>&& lock, const TensorInfo::Ptr& memory, ITensorPtr newPtr =
//         nullptr);

//         operator ITensorPtr() const;
//         bool isValid() const;
//         void valid(bool b);

//     private:
//         std::unique_lock<std::mutex> lock;
//         TensorInfo::Ptr memory;
//         ITensorPtr newPtr;
//     };

//     SharedTensor::Ptr findOrCreate(const std::string& key, std::function<ITensorPtr(void)> create, bool valid =
//     true);

//     SharedTensor::Ptr get(const std::string& key) const;

// protected:
//     // mutable std::mutex guard;
//     std::unordered_map<size_t, TensorInfo::Ptr> m_storage;
// };

// }  // namespace one_plugin
}  // namespace ov

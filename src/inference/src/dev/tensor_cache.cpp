// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/runtime/tensor_cache.hpp"

#include <memory>

#include "openvino/runtime/system_conf.hpp"

namespace ov {
// namespace one_plugin {

std::string get_weights_id(const std::shared_ptr<ov::op::v0::Constant>& constant) {
    char ptr[32];
    snprintf(ptr, sizeof ptr, "%p", constant->get_data_ptr());
    return constant->get_friendly_name() + "_" + std::to_string(ov::shape_size(constant->get_shape())) + "_" + ptr;
}

std::multimap<std::string, WeightsCache::TensorInfo>::const_iterator WeightsCache::get(std::string id) const {
    return m_storage.find(id);
}
std::multimap<std::string, WeightsCache::TensorInfo>::const_iterator WeightsCache::findOrCreate(std::string id, std::function<WeightsCache::TensorInfo::Ptr(void)> create) {
    WeightsCache::TensorInfo::Ptr new_ptr;

    auto it = get(id);
    if (it == m_storage.end()) {
        new_ptr = create();
        if (new_ptr) {
            std::cout << "No tensor " << id << ". Added new tensor " << new_ptr->m_tensor.data() << ". Size: " << new_ptr->m_tensor.get_byte_size() << std::endl;
            it = m_storage.insert({id, *new_ptr.get()});
        }
    } else {
        std::cout << "Found tensor " << it->first << " " << it->second.m_tensor.data() << ". Size: " << it->second.m_tensor.get_byte_size() << std::endl;
    }
    return it;
}

std::multimap<std::string, WeightsCache::TensorInfo>::const_iterator WeightsCache::replaceWith(
        std::multimap<std::string, WeightsCache::TensorInfo>::const_iterator pos, WeightsCache::TensorInfo::Ptr item) {
    auto id = pos->first;
    m_storage.erase(pos);
    return m_storage.insert({id, *item.get()});
}

// WeightsCache::SharedTensor::SharedTensor(std::unique_lock<std::mutex>&& lock,
//                                          const TensorInfo::Ptr& memory,
//                                          ITensorPtr newPtr)
//     : lock(std::move(lock)),
//       memory(memory),
//       newPtr(newPtr) {}

// WeightsCache::SharedTensor::operator ITensorPtr() const {
//     return memory->shared_tensor;
// }

// bool WeightsCache::SharedTensor::isValid() const {
//     return memory->valid.load(std::memory_order_acquire);
// }

// void WeightsCache::SharedTensor::valid(bool b) {
//     memory->valid.store(b, std::memory_order_release);
// }

// WeightsCache::SharedTensor::Ptr WeightsCache::findOrCreate(const std::string& key,
//                                                            std::function<ITensorPtr(void)> create,
//                                                            bool valid) {
//     TensorInfo::Ptr ptr;
//     ITensorPtr newPtr;
//     {
//         std::unique_lock<std::mutex> lock(guard);
//         auto found = shared_tensors.find(key);

//         if (found == shared_tensors.end() || !((ptr = found->second) && (newPtr = ptr->shared_tensor))) {
//             newPtr = create();
//             ptr = std::make_shared<TensorInfo>(newPtr, valid);
//             shared_tensors[key] = ptr;
//         }
//     }
//     return std::make_shared<SharedTensor>(ptr->valid.load(std::memory_order_relaxed)
//                                               ? std::unique_lock<std::mutex>(ptr->guard, std::defer_lock)
//                                               : std::unique_lock<std::mutex>(ptr->guard),
//                                           ptr,
//                                           newPtr);
// }

// WeightsCache::SharedTensor::Ptr WeightsCache::get(const std::string& key) const {
//     TensorInfo::Ptr ptr;
//     ITensorPtr newPtr;
//     {
//         std::unique_lock<std::mutex> lock(guard);
//         auto found = shared_tensors.find(key);

//         if (found == shared_tensors.end() || !((ptr = found->second) && (newPtr = ptr->shared_tensor)))
//             OPENVINO_THROW("Unknown shared tensor with key ", key);
//     }
//     return std::make_shared<SharedTensor>(ptr->valid.load(std::memory_order_relaxed)
//                                               ? std::unique_lock<std::mutex>(ptr->guard, std::defer_lock)
//                                               : std::unique_lock<std::mutex>(ptr->guard),
//                                           ptr,
//                                           newPtr);
// }

// }  // namespace one_plugin
}  // namespace ov

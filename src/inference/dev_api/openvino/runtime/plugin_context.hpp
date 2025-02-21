// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/runtime/tensor_cache.hpp"
#include "openvino/runtime/common.hpp"

namespace ov {
// namespace one_plugin {

class OPENVINO_RUNTIME_API PluginContext {
public:
    typedef std::shared_ptr<PluginContext> Ptr;
    typedef std::shared_ptr<const PluginContext> CPtr;

    PluginContext(WeightsCache::Ptr tensor_cache);

    WeightsCache::Ptr get_tensor_cache() const {
        return m_tensor_cache;
    }

private:
    WeightsCache::Ptr m_tensor_cache;
};

// }  // namespace one_plugin
}  // namespace ov

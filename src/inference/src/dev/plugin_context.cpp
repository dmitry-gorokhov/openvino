// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "openvino/runtime/plugin_context.hpp"

namespace ov {
// namespace one_plugin {

PluginContext::PluginContext(WeightsCache::Ptr tensor_cache) : m_tensor_cache(tensor_cache) {}


// }  // namespace one_plugin
}  // namespace ov

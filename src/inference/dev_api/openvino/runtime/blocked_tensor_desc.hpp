// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "itensor_desc.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
// namespace one_plugin {

class OPENVINO_RUNTIME_API BlockedTensorDesc : public virtual ITensorDesc {
public:
    typedef std::shared_ptr<BlockedTensorDesc> Ptr;
    typedef std::shared_ptr<const BlockedTensorDesc> CPtr;

public:
    BlockedTensorDesc() = default;
    BlockedTensorDesc(const ov::Shape& shape);
    BlockedTensorDesc(const ov::Shape& blocked_dims,
                      const ov::Shape& order,
                      const ov::Shape& strides = {},
                      const ov::Shape& offset_padding_to_data = {});
    ~BlockedTensorDesc() override = default;

    const ov::Shape& get_blocked_dims() const;
    const ov::Shape& get_order() const;
    const ov::Shape& get_strides() const;
    const ov::Shape& get_offset_padding_to_data() const;

protected:
    mutable ov::Shape m_blocked_dims;
    mutable ov::Shape m_order;
    mutable ov::Shape m_strides;
    mutable ov::Shape m_offset_padding_to_data;
};

// }  // namespace one_plugin
}  // namespace ov
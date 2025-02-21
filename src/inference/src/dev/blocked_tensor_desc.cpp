#include "openvino/runtime/blocked_tensor_desc.hpp"

namespace ov {
// namespace one_plugin {

static ov::Shape make_range(size_t size) {
    ov::Shape vec(size, 0);
    std::iota(vec.begin(), vec.end(), 0);
    return vec;
}

BlockedTensorDesc::BlockedTensorDesc(const ov::Shape& shape) : BlockedTensorDesc(shape, make_range(shape.size())) {}

BlockedTensorDesc::BlockedTensorDesc(const ov::Shape& blocked_dims,
                                     const ov::Shape& order,
                                     const ov::Shape& strides,
                                     const ov::Shape& offset_padding_to_data)
    : m_blocked_dims(blocked_dims),
      m_order(order) {
    if (strides.empty() && !order.empty()) {
        if (std::any_of(blocked_dims.begin(), blocked_dims.end(), [](size_t dim) {
                return dim == 0;
            })) {
            m_strides.resize(order.size(), 0);
        } else {
            m_strides.resize(order.size(), 1);
            for (size_t i = 2; i <= order.size(); i++) {
                m_strides[order.size() - i] =
                    m_strides[order.size() - (i - 1)] * m_blocked_dims[blocked_dims.size() - (i - 1)];
            }
        }
    } else {
        m_strides = strides;
    }

    if (offset_padding_to_data.empty() && !order.empty()) {
        m_offset_padding_to_data.resize(order.size(), 0);
    } else {
        m_offset_padding_to_data = offset_padding_to_data;
    }

    auto rank = m_blocked_dims.size();
    OPENVINO_ASSERT(m_order.size() == rank && m_strides.size() == rank && m_offset_padding_to_data.size() == rank,
                    "Order, blocked dims, offset padding to data and strides must have equals size");
}

const ov::Shape& BlockedTensorDesc::get_blocked_dims() const {
    return m_blocked_dims;
}
const ov::Shape& BlockedTensorDesc::get_order() const {
    return m_order;
}
const ov::Shape& BlockedTensorDesc::get_strides() const {
    return m_strides;
}
const ov::Shape& BlockedTensorDesc::get_offset_padding_to_data() const {
    return m_offset_padding_to_data;
}

// }  // namespace one_plugin
}  // namespace ov

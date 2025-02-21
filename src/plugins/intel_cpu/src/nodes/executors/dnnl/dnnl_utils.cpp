// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "nodes/executors/dnnl/dnnl_utils.hpp"

#include <common/primitive_desc_iface.hpp>
#include <oneapi/dnnl/dnnl.hpp>

#include "cpu_memory.h"
#include "memory_desc/cpu_memory_desc_utils.h"
#include "memory_desc/dnnl_memory_desc.h"
#include "nodes/executors/executor.hpp"
#include "nodes/reorder.h"
#include "utils/cpu_utils.hpp"

namespace ov::intel_cpu::utils {

MemoryPtr prepareWeightsMemory(const DnnlMemoryDescPtr srcWeightDesc,
                               const DnnlMemoryDescPtr dstWeightDesc,
                               const MemoryCPtr weightsMem,
                               const ExecutorContext::CPtr context,
                               const bool needShiftSignedToUnsigned) {
    const auto& eng = context->getEngine();
    const auto& format = dstWeightDesc->serializeFormat();

    auto create = [&]() {
        // https://oneapi-src.github.io/oneDNN/dev_guide_int8_computations.html?highlight=128#inputs-of-the-same-type-s8
        auto src_wdt = srcWeightDesc->getPrecision();
        auto dst_wdt = dstWeightDesc->getPrecision();
        if (needShiftSignedToUnsigned && src_wdt.is_integral_number() && src_wdt.is_signed() &&
            dst_wdt.is_integral_number() && !dst_wdt.is_signed()) {
            assert(src_wdt.bitwidth() == dst_wdt.bitwidth());

            // prevent reorderData from doing conversion
            Memory srcMemory{eng, srcWeightDesc->cloneWithNewPrecision(dst_wdt), weightsMem->getData()};
            MemoryPtr _ptr = std::make_shared<Memory>(eng, dstWeightDesc);
            auto rtCache = context->getRuntimeCache();
            node::Reorder::reorderData(srcMemory, *_ptr, rtCache);

            // do shift
            auto count = _ptr->getSize() / _ptr->getDesc().getPrecision().size();
            if (dst_wdt == ov::element::u8) {
                auto* data = _ptr->getDataAs<uint8_t>();
                for (size_t i = 0; i < count; i++) {
                    data[i] = data[i] + 128;
                }
            } else if (dst_wdt == ov::element::u4) {
                auto* data = _ptr->getDataAs<uint8_t>();
                for (size_t i = 0; i < count; i++) {
                    auto low = (data[i] & 0xF) + 8;
                    auto high = (data[i] >> 4) + 8;
                    data[i] = (high << 4) | (low & 0xF);
                }
            } else {
                OPENVINO_ASSERT(false, "Unsupported data type for shiftting sign to unsign");
            }
            return _ptr;
        }

        Memory srcMemory{eng, srcWeightDesc, weightsMem->getData()};
        MemoryPtr _ptr = std::make_shared<Memory>(eng, dstWeightDesc);
        auto rtCache = context->getRuntimeCache();
        node::Reorder::reorderData(srcMemory, *_ptr, rtCache);

        return _ptr;
    };

    std::cout << "weightsMem->getIs();" <<  weightsMem->getId() << std::endl;

    if (const auto tensorCache = context->getTensorCache()) {
        auto weightsId = weightsMem->getId();
        if (!weightsId.empty()) {
            auto it = tensorCache->get(weightsId);
            if (it != tensorCache->m_storage.end()) {
                const auto& tensorInfo = it->second;
                const auto& t = tensorInfo.m_tensor;
                const auto& td = tensorInfo.m_tensor_desc;

                Shape newShape(t.get_shape().empty() ? ov::Shape(1, 1) : t.get_shape());
                CpuBlockedMemoryDesc memDesc(t.get_element_type(), newShape, td->get_blocked_dims(),
                    td->get_order(), 0, td->get_offset_padding_to_data(), td->get_strides());
                Memory srcMemory{eng, memDesc, t.data()};

                ov::Shape dstShape(dstWeightDesc->getShape().getStaticDims());
                ov::Tensor dstTensor(dstWeightDesc->getPrecision(), dstShape);

                MemoryPtr ptr = std::make_shared<Memory>(eng, dstWeightDesc, dstTensor.data());
                auto rtCache = context->getRuntimeCache();
                node::Reorder::reorderData(srcMemory, *ptr, rtCache);

                // const auto& memDesc = memPtr->getDescPtr()->as<CpuBlockedMemoryDesc>();
                const auto& new_memDesc = ptr->getDescPtr()->as<BlockedMemoryDesc>();
                // ov::Tensor tensor(ptr->getPrecision(), ptr->getStaticDims(), ptr->getData());
                auto desc = std::make_shared<const ov::BlockedTensorDesc>(new_memDesc->getBlockDims(), new_memDesc->getOrder(),
                    new_memDesc->getStrides(), new_memDesc->getOffsetPaddingToData());
                auto new_ti = std::make_shared<ov::WeightsCache::TensorInfo>(dstTensor, desc);

                // tensorCache->m_storage.insert({weightsId, *new_ti.get()});
                std::cout << "For weights with id " << weightsId << " replace tensor " << it->first << std::endl;
                tensorCache->replaceWith(it, new_ti);

                // const auto privateWeightCache = context->getPrivateWeighCache();
                // (*privateWeightCache)[format] = ptr;

                return ptr;
            }
        }
    }

    const auto privateWeightCache = context->getPrivateWeighCache();
    OPENVINO_ASSERT(privateWeightCache, "privateWeightCache is nullptr");
    if (privateWeightCache) {
        auto itr = privateWeightCache->find(format);
        if (privateWeightCache->end() != itr) {
            return itr->second;
        }
    }

    auto globalWeightCache = context->getWeightsCache();
    MemoryPtr ptr;
    if (globalWeightCache && dnnl::memory::format_kind::blocked == dstWeightDesc->getDnnlDesc().get_format_kind()) {
        ptr = *globalWeightCache->findOrCreate(DnnlExtensionUtils::computeWeightsStringHash(weightsMem, dstWeightDesc),
                                               create);
    } else {
        ptr = create();
    }

    (*privateWeightCache)[format] = ptr;

    return ptr;
}

}  // namespace ov::intel_cpu::utils

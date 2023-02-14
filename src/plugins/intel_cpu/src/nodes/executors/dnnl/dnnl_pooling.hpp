// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: remove relative path
#include "../pooling.hpp"
#include "dnnl.hpp"
#include "dnnl_descriptor.h"
#include "memory_desc/dnnl_blocked_memory_desc.h"

namespace ov {
namespace intel_cpu {



class DnnlPoolingExecutor : public PoolingExecutor {
public:
    DnnlPoolingExecutor(const ExecutorContext::CPtr context);

    bool init(const PoolingAttrs& poolingAttrs,
              const std::vector<MemoryDescPtr>& srcDescs,
              const std::vector<MemoryDescPtr>& dstDescs,
              const dnnl::primitive_attr &attr) override;
    void exec(const std::vector<MemoryCPtr>& src,
              const std::vector<MemoryPtr>& dst,
              std::unordered_map<int, MemoryPtr> postOpsArgs) override;

    impl_desc_type getImplType() const override {
        return implType;
    }

    static DnnlDesriptor createDescriptor(const PoolingAttrs& poolingAttrs,
                                          const std::vector<MemoryDescPtr>& srcDescs,
                                          const std::vector<MemoryDescPtr>& dstDescs) {
        /*auto inputShape0 = srcDescs[0]->getShape();
        const VectorDims inStrides0 = getStridesAndModifyShape(inputShape0, matmulAttrs.transposeA);
        auto inDataDesc0 = std::make_shared<DnnlBlockedMemoryDesc>(srcDescs[0]->getPrecision(), inputShape0, inStrides0);

        auto inputShape1 = srcDescs[1]->getShape();
        const VectorDims inStrides1 = getStridesAndModifyShape(inputShape1, matmulAttrs.transposeB);
        auto inDataDesc1 = std::make_shared<DnnlBlockedMemoryDesc>(srcDescs[1]->getPrecision(), inputShape1, inStrides1);

        auto outputShape = dstDescs[0]->getShape();
        auto outDataDesc = std::make_shared<DnnlBlockedMemoryDesc>(dstDescs[0]->getPrecision(), outputShape);

        std::shared_ptr<dnnl::matmul::desc> matmul_desc;
        if (matmulAttrs.withBias) {
            // oneDNN matmul requires shape for bias desc to be the same rank
            VectorDims biasDims(outputShape.getRank(), 1);
            const auto outDims = outputShape.getStaticDims();
            const auto chIdx = outputShape.getRank() - 1;
            biasDims[chIdx] = outDims[chIdx];
            const auto bdt = DnnlExtensionUtils::IEPrecisionToDataType(srcDescs[2]->getPrecision());
            auto biasDesc = dnnl::memory::desc(DnnlExtensionUtils::convertToDnnlDims(biasDims), bdt, dnnl::memory::format_tag::any);

            matmul_desc.reset(new dnnl::matmul::desc(inDataDesc0->getDnnlDesc(),
                                                     inDataDesc1->getDnnlDesc(),
                                                     biasDesc,
                                                     outDataDesc->getDnnlDesc()));
        } else {
            matmul_desc.reset(new dnnl::matmul::desc(inDataDesc0->getDnnlDesc(),
                                                     inDataDesc1->getDnnlDesc(),
                                                     outDataDesc->getDnnlDesc()));
        }*/

        std::shared_ptr<dnnl::pooling_v2_forward::desc> pooling_desc;

        return DnnlDesriptor(pooling_desc);
    }


    struct Key {
        PoolingAttrs poolingAttrs;
        DnnlMemoryDescPtr inp0;
        DnnlMemoryDescPtr inp1;
        DnnlMemoryDescPtr bias;
        DnnlMemoryDescPtr out;
        dnnl::primitive_attr attr;

        Key(const PoolingAttrs& poolingAttrs,
            const std::vector<MemoryDescPtr>& srcDescs,
            const std::vector<MemoryDescPtr>& dstDescs,
            const dnnl::primitive_attr &attr);
        size_t hash() const;
        bool operator==(const Key& rhs) const;
    };

private:
    //static std::pair<Shape, Shape> makeDummyInputShapes(const MatMulAttrs& matmulAttrs, const Shape& in0, const Shape& in1);

    dnnl::stream stream;

    PoolingAttrs poolingAttrs;
    std::shared_ptr<dnnl::pooling_v2_forward> prim;
    MemoryPtr scratchpadMemory;
    impl_desc_type implType = impl_desc_type::undef;
};

class DnnlPoolingExecutorBuilder : public PoolingExecutorBuilder {
public:
    bool isSupported(const PoolingAttrs& poolingAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs,
                     const dnnl::primitive_attr &attr) const override {
        // TODO: add correct conditions
        return true;
    }

    PoolingExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<DnnlPoolingExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov
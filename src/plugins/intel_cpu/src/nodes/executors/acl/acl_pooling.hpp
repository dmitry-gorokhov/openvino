// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: remove relative path
#include "../pooling.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"

namespace ov {
namespace intel_cpu {

class AclPoolingExecutor : public PoolingExecutor {
public:
    AclPoolingExecutor(const ExecutorContext::CPtr context);

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

private:
    PoolingAttrs poolingAttrs;
    impl_desc_type implType = impl_desc_type::gemm_acl;

    arm_compute::Tensor srcTensor;
    arm_compute::Tensor dstTensor;
    arm_compute::Tensor dst1Tensor;
    std::unique_ptr<arm_compute::NEPoolingLayer> pooling = nullptr;
};

class AclPoolingExecutorBuilder : public PoolingExecutorBuilder {
public:
    bool isSupported(const PoolingAttrs& poolingAttrs,
                     const std::vector<MemoryDescPtr>& srcDescs,
                     const std::vector<MemoryDescPtr>& dstDescs) const override {
        if ((srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP32 &&
             dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP32) &&
            (srcDescs[0]->getPrecision() != InferenceEngine::Precision::FP16 &&
             dstDescs[0]->getPrecision() != InferenceEngine::Precision::FP16)) {
            std::cout << "AclPoolingExecutorBuilder::isSupported - presicion is not supported: src=" <<
             srcDescs[0]->getPrecision() << "src=" << dstDescs[0]->getPrecision() << std::endl;
            return false;
        }

        if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
              dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
              dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
            std::cout << "AclPoolingExecutorBuilder::isSupported - layout is not supported" << std::endl;
              return false;
        }

        return true;
    }

    PoolingExecutorPtr makeExecutor(const ExecutorContext::CPtr context) const override {
        return std::make_shared<AclPoolingExecutor>(context);
    }
};

}   // namespace intel_cpu
}   // namespace ov
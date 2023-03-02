// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// TODO: remove relative path
#include "../pooling.hpp"
#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "utils/debug_capabilities.h"

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
    arm_compute::Tensor indTensor;
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
            DEBUG_LOG("AclPoolingExecutor does not support precisions: input precision=",
                      srcDescs[0]->getPrecision(), " output precision=", dstDescs[0]->getPrecision());
            return false;
        }

        if (!(srcDescs[0]->hasLayoutType(LayoutType::ncsp) &&
              dstDescs[0]->hasLayoutType(LayoutType::ncsp)) &&
            !(srcDescs[0]->hasLayoutType(LayoutType::nspc) &&
              dstDescs[0]->hasLayoutType(LayoutType::nspc))) {
                DEBUG_LOG("AclPoolingExecutor does not support such layouts");
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
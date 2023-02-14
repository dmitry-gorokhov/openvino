// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_pooling.hpp"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

inline arm_compute::TensorShape shapeCast(const VectorDims& dims) {
    arm_compute::TensorShape tensorShape;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        tensorShape.set(dims.size() - i - 1, dims[i], false);
    }
    if (tensorShape.num_dimensions() == 0) {
        tensorShape.set(0, 1, false);
        tensorShape.set_num_dimensions(1);
    }
    return tensorShape;
}

inline arm_compute::DataType precisionToAclDataType(InferenceEngine::Precision precision) {
    switch (precision) {
        case InferenceEngine::Precision::I8:    return arm_compute::DataType::S8;
        case InferenceEngine::Precision::U8:    return arm_compute::DataType::U8;
        case InferenceEngine::Precision::I16:   return arm_compute::DataType::S16;
        case InferenceEngine::Precision::U16:   return arm_compute::DataType::U16;
        case InferenceEngine::Precision::I32:   return arm_compute::DataType::S32;
        case InferenceEngine::Precision::U32:   return arm_compute::DataType::U32;
        case InferenceEngine::Precision::FP16:  return arm_compute::DataType::F16;
        case InferenceEngine::Precision::FP32:  return arm_compute::DataType::F32;
        case InferenceEngine::Precision::FP64:  return arm_compute::DataType::F64;
        case InferenceEngine::Precision::I64:   return arm_compute::DataType::S64;
        case InferenceEngine::Precision::BF16:  return arm_compute::DataType::BFLOAT16;
        default:                                return arm_compute::DataType::UNKNOWN;
    }
}

inline arm_compute::DataLayout getAclDataLayoutByMemoryDesc(MemoryDescPtr desc) {
    if (desc->hasLayoutType(LayoutType::ncsp)) {
        if (desc->getShape().getRank() == 4) return arm_compute::DataLayout::NCHW;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NCDHW;
    } else if (desc->hasLayoutType(LayoutType::nspc)) {
        if (desc->getShape().getRank() == 4) return arm_compute::DataLayout::NHWC;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NDHWC;
    }
    return arm_compute::DataLayout::UNKNOWN;
}

AclPoolingExecutor::AclPoolingExecutor(const ExecutorContext::CPtr context) : PoolingExecutor(context) {}

bool AclPoolingExecutor::init(const PoolingAttrs& poolingAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) {
    std::cout << "AclPoolingExecutor::init" << std::endl;
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();

    if (srcDims.size() != 4) {
        std::cout << "AclPoolingExecutor::init only 4D input tensors are supported. Tensor rank: " << srcDims.size() << std::endl;
        return false;
    }
    TensorInfo srcTensorInfo = TensorInfo(shapeCast(srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0]));
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0]));

    arm_compute::PoolingLayerInfo pool_info;
    unsigned int pad_left   = poolingAttrs.data_pad_begin[1];
    unsigned int pad_right  = poolingAttrs.data_pad_end[1];
    unsigned int pad_top    = poolingAttrs.data_pad_begin[0];
    unsigned int pad_bottom = poolingAttrs.data_pad_end[0];
    unsigned int kernel_w   = poolingAttrs.kernel[1];
    unsigned int kernel_h   = poolingAttrs.kernel[0];
    unsigned int stride_x   = poolingAttrs.stride[1];
    unsigned int stride_y   = poolingAttrs.stride[0];

    //TODO: need to fix
    arm_compute::DimensionRoundingType round = arm_compute::DimensionRoundingType::CEIL;

    pool_info.data_layout       = getAclDataLayoutByMemoryDesc(srcDescs[0]);
    pool_info.pool_size         = arm_compute::Size2D(kernel_w, kernel_h);
    pool_info.pad_stride_info   = arm_compute::PadStrideInfo(stride_x, stride_y, pad_left, pad_right, pad_top, pad_bottom, round);
    pool_info.exclude_padding = poolingAttrs.exclude_pad;

    if (poolingAttrs.algorithm == Algorithm::PoolingMax) {
        pool_info.pool_type = arm_compute::PoolingType::MAX;
    } else if (poolingAttrs.algorithm == Algorithm::PoolingAvg) {
        pool_info.pool_type = arm_compute::PoolingType::AVG;
    } else {
        return false;
    }

    if (!arm_compute::NEPoolingLayer::validate(&srcTensorInfo, &dstTensorInfo, pool_info)) {
        std::cout << "AclPoolingExecutor::init validate fails" << std::endl;
        return false;
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    pooling = std::make_unique<arm_compute::NEPoolingLayer>();
    if (/*poolingAttrs.algorithm == Algorithm::PoolingMax*/dstDescs.size() > 1) {
auto dst1Dims = dstDescs[1]->getShape().getStaticDims();
TensorInfo dst1TensorInfo = TensorInfo(shapeCast(dst1Dims), 1,
    precisionToAclDataType(dstDescs[1]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[1]));
        dst1Tensor.allocator()->init(dst1TensorInfo);
        pooling->configure(&srcTensor, &dstTensor, pool_info, &dst1Tensor);
        std::cout << "INDICES!" << std::endl;
    } else {
        pooling->configure(&srcTensor, &dstTensor, pool_info);
    }

    std::cout << "AclPoolingExecutor::init OK" << std::endl;
    return true;
}

void AclPoolingExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());
    if (dst.size() > 1) dst1Tensor.allocator()->import_memory(dst[1]->GetPtr());

    pooling->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
    if (dst.size() > 1) dst1Tensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov

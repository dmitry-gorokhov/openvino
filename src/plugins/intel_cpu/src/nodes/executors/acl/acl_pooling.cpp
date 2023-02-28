// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "acl_pooling.hpp"
#include "acl_utils.hpp"
#include "arm_compute/core/utils/misc/ShapeCalculator.h"

namespace ov {
namespace intel_cpu {

using namespace arm_compute;

AclPoolingExecutor::AclPoolingExecutor(const ExecutorContext::CPtr context) : PoolingExecutor(context) {}

bool AclPoolingExecutor::init(const PoolingAttrs& poolingAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) {
    auto srcDims = srcDescs[0]->getShape().getStaticDims();
    auto dstDims = dstDescs[0]->getShape().getStaticDims();
    /*if (poolingAttrs.dilation != ov::Strides{1, 1}) {
        std::cout << "AclPoolingExecutor::init unsupported dilation!" << std::endl;
    }*/

    if (srcDims.size() != 4) {
        std::cout << "AclPoolingExecutor::init only 4D input tensors are supported. Tensor rank: " << srcDims.size() << std::endl;
        //return false;
    }

    /*VectorDims srcDimsReduced;
    VectorDims dstDimsReduced;
    if (srcDims.size() == 5) {
        srcDimsReduced.push_back(srcDims[0] * srcDims[1]);
        srcDimsReduced.push_back(srcDims[2]);
        srcDimsReduced.push_back(srcDims[3]);
        srcDimsReduced.push_back(srcDims[4]);
        dstDimsReduced.push_back(dstDims[0] * dstDims[1]);
        dstDimsReduced.push_back(dstDims[2]);
        dstDimsReduced.push_back(dstDims[3]);
        dstDimsReduced.push_back(dstDims[4]);
    }*/

    TensorInfo srcTensorInfo = TensorInfo(shapeCast(/*(srcDims.size() == 5) ? srcDimsReduced :*/ srcDims), 1,
    precisionToAclDataType(srcDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(srcDescs[0])/*arm_compute::DataLayout::NCHW*/);
    TensorInfo dstTensorInfo = TensorInfo(shapeCast(/*(srcDims.size() == 5) ? dstDimsReduced :*/ dstDims), 1,
    precisionToAclDataType(dstDescs[0]->getPrecision()), getAclDataLayoutByMemoryDesc(dstDescs[0])/*arm_compute::DataLayout::NCHW*/;


    arm_compute::PoolingLayerInfo pool_info;
    unsigned int pad_left   = poolingAttrs.data_pad_begin[1];
    unsigned int pad_right  = poolingAttrs.data_pad_end[1];
    unsigned int pad_top    = poolingAttrs.data_pad_begin[0];
    unsigned int pad_bottom = poolingAttrs.data_pad_end[0];
    unsigned int kernel_w   = poolingAttrs.kernel[1];
    unsigned int kernel_h   = poolingAttrs.kernel[0];
    unsigned int stride_x   = poolingAttrs.stride[1];
    unsigned int stride_y   = poolingAttrs.stride[0];

    arm_compute::DimensionRoundingType round = (poolingAttrs.rounding == op::RoundingType::CEIL) ?
                                                arm_compute::DimensionRoundingType::CEIL : arm_compute::DimensionRoundingType::FLOOR;

    pool_info.data_layout       = arm_compute::DataLayout::NCHW;//getAclDataLayoutByMemoryDesc(srcDescs[0]);
    pool_info.pool_size         = arm_compute::Size2D(kernel_w, kernel_h);
    pool_info.pad_stride_info   = arm_compute::PadStrideInfo(stride_x, stride_y, pad_left, pad_right, pad_top, pad_bottom, round);
    //pool_info.is_global_pooling = false;

    if (poolingAttrs.algorithm == Algorithm::PoolingMax) {
        pool_info.pool_type = arm_compute::PoolingType::MAX;
        pool_info.exclude_padding = (poolingAttrs.pad_type != op::PadType::EXPLICIT);
    } else if (poolingAttrs.algorithm == Algorithm::PoolingAvg) {
        pool_info.pool_type = arm_compute::PoolingType::AVG;
        pool_info.exclude_padding = poolingAttrs.exclude_pad;
    } else {
        return false;
    }

    /*arm_compute::TensorInfo ti = arm_compute::TensorInfo(arm_compute::misc::shape_calculator::compute_pool_shape(srcTensorInfo, pool_info),
     1, dstTensorInfo.data_type());*/

    TensorInfo indTensorInfo;
    if (dstDescs.size() > 1) {
        std::cout << "AclPoolingExecutor::init - indices branch" << std::endl;
        auto indDims = dstDescs[1]->getShape().getStaticDims();
        indTensorInfo = TensorInfo(shapeCast(indDims), 1, arm_compute::DataType::U32, getAclDataLayoutByMemoryDesc(srcDescs[0]));
        arm_compute::Status s = arm_compute::NEPoolingLayer::validate(&srcTensorInfo, &dstTensorInfo, pool_info, &indTensorInfo);
        if (!s) {
            std::cout << "validate failed (ind): " << s.error_description() << std::endl;
            return false;
        }
    } else {
        std::cout << "AclPoolingExecutor::init - no indices branch" << std::endl;
        arm_compute::Status s = arm_compute::NEPoolingLayer::validate(&srcTensorInfo, &dstTensorInfo, pool_info);
        if (!s) {
            std::cout << "validate failed (no ind): " << s.error_description() << std::endl;
            return false;
        }
    }

    srcTensor.allocator()->init(srcTensorInfo);
    dstTensor.allocator()->init(dstTensorInfo);

    pooling = std::make_unique<arm_compute::NEPoolingLayer>();
    if (dstDescs.size() > 1) {
        indTensor.allocator()->init(indTensorInfo);
        pooling->configure(&srcTensor, &dstTensor, pool_info, &indTensor);
        std::cout << "INDICES!" << std::endl;
    } else {
        pooling->configure(&srcTensor, &dstTensor, pool_info);
    }

    return true;
}

void AclPoolingExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    srcTensor.allocator()->import_memory(src[0]->GetPtr());
    dstTensor.allocator()->import_memory(dst[0]->GetPtr());
    if (dst.size() > 1) indTensor.allocator()->import_memory(dst[1]->GetPtr());

    pooling->run();

    srcTensor.allocator()->free();
    dstTensor.allocator()->free();
    if (dst.size() > 1) indTensor.allocator()->free();
}

}   // namespace intel_cpu
}   // namespace ov

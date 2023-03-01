// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

namespace ov {
namespace intel_cpu {

inline VectorDims dimsCast(const VectorDims& dims, size_t requiredSize) {
    if (dims.size() == requiredSize)
        return dims;
    VectorDims returnDims;
    std::cout << "dimsCast: Original VectorDims: "; for (auto i : dims) std::cout << i << " "; std::cout << std::endl;
    if (dims.size() > requiredSize) {
        Dim dim = dims[0];
        for (int i = 1; i < dims.size() - requiredSize + 1; i++) {
            dim *= dims[i];
        }
        returnDims.push_back(dim);
        for (int i = dims.size() - requiredSize + 1; i < dims.size(); i++) {
            returnDims.push_back(dims[i]);
        }
    } else {
        returnDims = dims;
        for (int i = dims.size(); i < requiredSize; i++) {
            returnDims.push_back(1);
        }
    }
    std::cout << "dimsCast: Final VectorDims: "; for (auto i : returnDims) std::cout << i << " "; std::cout << std::endl;
    return returnDims;
}

inline arm_compute::TensorShape shapeCast(const VectorDims& dims) {
    arm_compute::TensorShape tensorShape;
    std::cout << "shapeCast: after creation VectorDims: "; for (auto i : tensorShape) std::cout << i << " "; std::cout << std::endl;
    std::cout << "shapeCast: Original VectorDims: "; for (auto i : dims) std::cout << i << " "; std::cout << std::endl;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        tensorShape.set(dims.size() - i - 1, dims[i], false);
        std::cout << "shapeCast: intermidiate VectorDims: "; for (auto i : tensorShape) std::cout << i << " "; std::cout << std::endl;
    }
    if (tensorShape.num_dimensions() == 0) {
        tensorShape.set(0, 1, false);
        tensorShape.set_num_dimensions(1);
    }
    std::cout << "shapeCast: Final VectorDims: "; for (auto i : tensorShape) std::cout << i << " "; std::cout << std::endl;
    return tensorShape;
}

/**
* @brief Return ComputeLibrary DataType that corresponds to the given precision
* @param precision precision to be converted
* @return ComputeLibrary DataType or UNKNOWN if precision is not mapped to DataType
*/
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

/**
* @brief Return ComputeLibrary DataLayout that corresponds to MemoryDecs layout
* @param desc MemoryDecs from which layout is retrieved
* @return ComputeLibrary DataLayout or UNKNOWN if MemoryDecs layout is not mapped to DataLayout
*/
inline arm_compute::DataLayout getAclDataLayoutByMemoryDesc(MemoryDescCPtr desc) {
    if (desc->hasLayoutType(LayoutType::ncsp)) {
        if (desc->getShape().getRank() == 4) return arm_compute::DataLayout::NCHW;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NCDHW; 
    } else if(desc->hasLayoutType(LayoutType::nspc)) {
        if (desc->getShape().getRank() == 4) return arm_compute::DataLayout::NHWC;
        if (desc->getShape().getRank() == 5) return arm_compute::DataLayout::NDHWC;
    }
    return arm_compute::DataLayout::UNKNOWN;
}

}   // namespace intel_cpu
}   // namespace ov

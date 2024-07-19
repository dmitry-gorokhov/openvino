// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "tpp_fullyconnected.hpp"

#include "nodes/executors/executor.hpp"
#include "nodes/executors/memory_arguments.hpp"
#include "utils/debug_capabilities.h"
#include "openvino/core/parallel.hpp"

namespace ov {
namespace intel_cpu {

bool TPPFCExecutor::supports(const FCConfig& config) {
    if (!config.postOps.empty()) {
        DEBUG_LOG("TPPFCExecutor: PostOps are not supported");
        return false;
    }

    const auto& srcDesc = config.descs.at(ARG_SRC);
    const auto& weiDesc = config.descs.at(ARG_WEI);
    const auto& dstDesc = config.descs.at(ARG_DST);
    if (!everyone_is(ov::element::f32, srcDesc->getPrecision(), dstDesc->getPrecision())) {
        DEBUG_LOG("TPPFCExecutor: supports only f32 src and dst precisions");
        return false;
    }

    if (!everyone_is(ov::element::f4e2m1, weiDesc->getPrecision())) {
        DEBUG_LOG("TPPFCExecutor: supports only f32 src and dst precisions");
        return false;
    }

    if (config.attrs.decompressionSubtractPtr) {
        DEBUG_LOG("TPPFCExecutor: doesn't support decompression subtract");
        return false;
    }

    if (config.attrs.decompressionMultiplyPtr && config.attrs.decompressionMultiplyPtr->getPrecision() != ov::element::f8e8m0) {
        DEBUG_LOG("TPPFCExecutor: supports only f8e8m0 decompression scales precision");
        return false;
    }

    if (config.attrs.withBias) {
        DEBUG_LOG("TPPFCExecutor: bias is not supported");
        return false;
        // const auto& biaDesc = config.descs.at(ARG_BIAS);
        // if (biaDesc->getPrecision() != ov::element::f32) {
        //     DEBUG_LOG("TPPFCExecutor: supports only f32 bias");
        //     return false;
        // }

        // const auto& biasDims = biaDesc->getShape().getStaticDims();
        // const auto& outDims = dstDesc->getShape().getDims();
        // const bool isByChannel = biasDims.back() == outDims.back();
        // if (!isByChannel || !std::all_of(biasDims.begin(), biasDims.end() - 1, [](const Dim dim) { return dim == 1; })) {
        //     DEBUG_LOG("TPPFCExecutor: only 'by channel' bias is supported");
        //     return false;
        // }
    }

    return true;
}

TPPFCExecutor::TPPFCExecutor(const FCAttrs& attrs,
                             const PostOps& postOps,
                             const MemoryArgs& memory,
                             const ExecutorContext::CPtr context) : m_attrs(attrs) {
    // const auto& srcDesc = memory.at(ARG_SRC)->getDescPtr();
    // const auto& weiDesc = memory.at(ARG_WEI)->getDescPtr();
    // const auto& dstDesc = memory.at(ARG_DST)->getDescPtr();

    // // Allocate TPP session
    // sess = TPPSession();

    // // Allocate TPP tensors
    // src = TPPTensor(sess, precisionToTPPDataType(srcDesc->getPrecision()), getTPPDataLayoutByMemoryDesc(srcDesc));
    // wei = TPPTensor(sess, precisionToTPPDataType(weiDesc->getPrecision()), getTPPDataLayoutByMemoryDesc(weiDesc, true),
    //                       weiDesc->getShape().getStaticDims());
    // dst = TPPTensor(sess, precisionToTPPDataType(dstDesc->getPrecision()), getTPPDataLayoutByMemoryDesc(dstDesc));

    // if (attrs.withBias) {
    //     const auto& biasDesc = memory.at(ARG_BIAS)->getDescPtr();
    //     bias = TPPTensor(sess, precisionToTPPDataType(biasDesc->getPrecision()), getTPPDataLayoutByMemoryDesc(biasDesc),
    //                            biasDesc->getShape().getStaticDims());
    //     with_bias = true;
    // } else {
    //     bias = TPPTensor(sess);
    // }

    // // Init FC params
    // params = TPPFCParams(sess, CSINN_RVV);

    // OPENVINO_ASSERT(csinn_fullyconnected_init(src.get(), dst.get(), wei.get(), bias.get(), params.get()) == CSINN_TRUE,
    //                 "TPPFCExecutor: failed to init FC");
}

bool TPPFCExecutor::update(const MemoryArgs& memory) {
    // // Weights and Bias have static shapes - no need to update them here
    // src = src.cloneWithNewShape(memory.at(ARG_SRC)->getDescPtr()->getShape().getStaticDims());
    // dst = dst.cloneWithNewShape(memory.at(ARG_DST)->getDescPtr()->getShape().getStaticDims());

    return true;
}

template <typename T>
static std::vector<T> normalizeDimsTo2D(const std::vector<T>& dims) {
    return {std::accumulate(dims.begin(), dims.end() - 1, (T)1, std::multiplies<T>()), dims[dims.size() - 1]};
}

static int8_t get_u4(const uint8_t& val, bool high) {
    return high ? (val >> 4) : (val & 0xF);
}

void TPPFCExecutor::execute(const MemoryArgs& memory) {
    auto src = memory.at(ARG_SRC);
    auto wei = memory.at(ARG_WEI);
    auto dst = memory.at(ARG_DST);

    auto psrc = src->getDataAs<const float>();
    auto pwei = wei->getDataAs<const uint8_t>();
    auto pdst = dst->getDataAs<float>();
    auto pscales = m_attrs.decompressionMultiplyPtr->getDataAs<const float8_e8m0>();

    auto srcDims = normalizeDimsTo2D(src->getDesc().getShape().getDims());
    auto weiDims = wei->getDesc().getShape().getDims();
    auto scalesShape = m_attrs.decompressionMultiplyPtr->getDesc().getShape().getDims();

    auto M = srcDims[0];
    auto K = srcDims[1];
    auto N = weiDims[0];
    auto kGroups = m_attrs.weightsNonTransposed ? scalesShape[0] : scalesShape[1];
    auto kGroupSize = K / kGroups;

    std::cerr << M << " " << K << " " << N << std::endl;
    std::cerr << scalesShape[0] << " " << scalesShape[1] << " " << scalesShape[2] << std::endl;

    // for (size_t m = 0; m < M; m++) {
    //     for (size_t n = 0; n < N; n++) {
    parallel_for2d(M, N, [&](size_t m, size_t n) {
            size_t dstIdx = m * N + n;
            pdst[dstIdx] = 0.f;

            for (size_t kb = 0; kb < kGroups; kb++) {
                size_t scalesIdx = m_attrs.weightsNonTransposed ? kb * N + n : n * kGroups + kb;
                auto fscale = static_cast<float>(pscales[scalesIdx]);

                for (size_t ki = 0; ki < kGroupSize; ki++) {
                    auto k = kb * kGroupSize + ki;
                    size_t srcIdx = m * K + k;
                    size_t weiIdx = m_attrs.weightsNonTransposed ? k * N + n : n * K + k;

                    auto fwei = static_cast<float>(float4_e2m1::from_bits(get_u4(pwei[weiIdx / 2], weiIdx % 2)));
                    pdst[dstIdx] += psrc[srcIdx] * (fwei * fscale);
                }
            }
    });
        // }
    // }

    // src.setData(memory.at(ARG_SRC)->getData());
    // wei.setData(memory.at(ARG_WEI)->getData());
    // dst.setData(memory.at(ARG_DST)->getData());
    // if (with_bias) {
    //     bias.setData(memory.at(ARG_BIAS)->getData());
    // }

    // OPENVINO_ASSERT(csinn_fullyconnected(src.get(), dst.get(), wei.get(), bias.get(), params.get()) == CSINN_TRUE,
    //                 "TPPFCExecutor: failed to execute");
}

void TPPFCExecutor::moveMemToNumaNode(int numaNodeID) {
    // if (curNumaNode == numaNodeID)
    //     return;
    // curNumaNode = numaNodeID;
    // mbind_move(packedWeights, numaNodeID);
    // if (m_attrs.withBias) {
    //     mbind_move(m_memoryArgs.at(ARG_BIAS), numaNodeID);
    // }
}

}  // namespace intel_cpu
}  // namespace ov
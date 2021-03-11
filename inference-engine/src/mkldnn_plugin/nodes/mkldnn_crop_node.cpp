// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mkldnn_crop_node.h"
#include <string>
#include <algorithm>
#include <mkldnn_types.h>
#include <mkldnn_extension_utils.h>
#include "ie_parallel.hpp"
#include "common/cpu_memcpy.h"
#include "utils/general_utils.h"

using namespace mkldnn;
using namespace MKLDNNPlugin;
using namespace InferenceEngine;

bool MKLDNNCropNode::isSupportedOperation(const std::shared_ptr<const ngraph::Node>& op, std::string& errorMessage) noexcept {
    try {
        auto stridedSlice = ngraph::as_type_ptr<const ngraph::op::v1::StridedSlice>(op);
        if (!stridedSlice) {
            errorMessage = "Node is not an instance of the StridedSlice operation.";
            return false;
        }
        auto beginNode = ngraph::as_type_ptr<ngraph::op::v0::Constant>(stridedSlice->get_input_node_shared_ptr(1));
        auto endNode = ngraph::as_type_ptr<ngraph::op::v0::Constant>(stridedSlice->get_input_node_shared_ptr(2));
        if (!beginNode || !endNode) {
            errorMessage = "Constant expected as the second and third inputs.";
            return false;
        }
        if (stridedSlice->get_input_size() > 3) {
            auto strideNode = ngraph::as_type_ptr<ngraph::op::v0::Constant>(stridedSlice->get_input_node_shared_ptr(3));
            if (!strideNode) {
                errorMessage = "Constant expected as the fourth input.";
                return false;
            }

            auto strideData = strideNode->cast_vector<int32_t>();
            for (auto & s : strideData) {
                if (s != 1) {
                    errorMessage = "Crop supports just a single stride.";
                    return false;
                }
            }
        }
    } catch (...) {
        return false;
    }
    return true;
}

MKLDNNCropNode::MKLDNNCropNode(const std::shared_ptr<ngraph::Node>& op, const mkldnn::engine& eng, MKLDNNWeightsSharing::Ptr &cache) :
        MKLDNNNode(op, eng, cache) {
    std::string errorMessage;
    if (!isSupportedOperation(op, errorMessage)) {
        THROW_IE_EXCEPTION_WITH_STATUS(NOT_IMPLEMENTED) << errorMessage;
    }

    auto stridedSlice = ngraph::as_type_ptr<const ngraph::op::v1::StridedSlice>(op);
    auto beginNode = ngraph::as_type_ptr<ngraph::op::v0::Constant>(stridedSlice->get_input_node_shared_ptr(1));

    auto beginData = beginNode->cast_vector<int64_t>();

    auto inputShape = stridedSlice->get_input_shape(0);

    auto convertToSet = [](const std::vector<int64_t>& mask) {
        std::set<size_t> axisSet{};
        for (size_t i = 0lu; i < static_cast<size_t>(mask.size()); ++i) {
            if (mask[i] == 1) {
                axisSet.emplace(i);
            }
        }
        return axisSet;
    };

    auto shrinkAxisMask = convertToSet(stridedSlice->get_shrink_axis_mask());
    auto newAxisMask = convertToSet(stridedSlice->get_new_axis_mask());
    auto ellipsisMask = convertToSet(stridedSlice->get_ellipsis_mask());
    auto beginMask = convertToSet(stridedSlice->get_begin_mask());

    std::vector<int64_t> axes, offsets;

    size_t inputShapeIdx = 0lu;
    uint64_t uniqId = 0;
    for (size_t axis = 0lu; axis < beginData.size(); ++axis) {
        // add dimensions hidden under the ellipsis mask if ellipsis mask is set
        if (ellipsisMask.count(axis)) {
            // only one bit in ellipsis mask is allowed
            int numNewAxisAfterEllipses = 0;
            int numInputAxisBeforeEllipses = 0;
            for (size_t i = 0lu; i < axis; ++i) {
                if (!newAxisMask.count(i))
                    numInputAxisBeforeEllipses++;
            }
            for (size_t i = axis + 1; i < beginData.size(); ++i) {
                if (newAxisMask.count(i))
                    numNewAxisAfterEllipses++;
            }

            // -1 because it's a position of ellipses
            size_t numInputAxisAfterEllipses = (beginData.size() - axis - numNewAxisAfterEllipses - 1);
            size_t numOfHiddenDims = inputShape.size() - numInputAxisAfterEllipses
                                               - numInputAxisBeforeEllipses;
            for (size_t i = 0; i < numOfHiddenDims; ++i) {
                axes.emplace_back(uniqId);
                uniqId++;
                offsets.emplace_back(0);

                inputShapeIdx++;
            }
        } else {
            // add new single dimension if newAxisMask is set
            if (newAxisMask.count(axis)) {
                offsets.emplace_back(0);
            } else if (shrinkAxisMask.count(axis)) {
                // skip this dimension if shrinkAxisMask is set (inputShapeIdx++)
                offsets.emplace_back(beginMask.count(axis) ? 0 : beginData[axis]);
                inputShapeIdx++;
            } else {
                int64_t lb = beginData[axis];
                // convert negative indexes to positive
                if (lb < 0)
                    lb = std::max(static_cast<int64_t>(inputShape[inputShapeIdx]) + lb,
                                  static_cast<int64_t>(0));
                // apply restrictions when beginData values more/less than max/min possible values.
                lb = std::min(static_cast<int64_t>(inputShape[inputShapeIdx]), lb);
                offsets.emplace_back(lb);
                inputShapeIdx++;
            }
            axes.emplace_back(uniqId);
            uniqId++;
        }
    }
    for (; inputShapeIdx < inputShape.size(); ++inputShapeIdx) {
        offsets.emplace_back(0);
        axes.emplace_back(uniqId);
        uniqId++;
    }

    auto outputShape = stridedSlice->get_output_shape(0);

    this->offsets.resize(static_cast<size_t>(outputShape.size()));  // plus one dim for batch
    this->dims.resize(static_cast<size_t>(outputShape.size()));     // plus one dim for batch
    for (int i = 0; i < outputShape.size(); i++)
        this->dims[i] = outputShape[i];

    for (int i = 0; i < axes.size(); i++) {
        this->offsets[axes[i]] = offsets[i];
    }

    channelAxis = 1;
    if (axes.size() == this->dims.size()) {
        for (size_t i = 0lu; i < axes.size(); i++) {
            if (axes[i] == 1) {
                channelAxis = static_cast<int>(i);
                break;
            }
        }
    }
}

void MKLDNNCropNode::getSupportedDescriptors() {
}

void MKLDNNCropNode::initSupportedPrimitiveDescriptors() {
    if (!supportedPrimitiveDescriptors.empty())
        return;

    InferenceEngine::Precision precision = getOriginalInputPrecisionAtPort(0);
    auto inputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    precision = getOriginalOutputPrecisionAtPort(0);
    auto outputDataType = MKLDNNExtensionUtils::IEPrecisionToDataType(precision);
    if (inputDataType != outputDataType) {
        outputDataType = inputDataType; // Crop doesn't convert precisions, only moves data
    }

    auto& inDims = getParentEdgeAt(0)->getDims();
    if (inDims.ndims() != 2 && inDims.ndims() != 4 && inDims.ndims() != 5) {
        THROW_IE_EXCEPTION << "Crop supports only 2d, 4d and 5d blobs.";
    }

    memory::format_tag fmt = memory::format_tag::undef;
    switch (inDims.ndims()) {
        case 2: fmt = memory::format_tag::nc; break;
        case 4: fmt = memory::format_tag::nchw; break;
        case 5: fmt = memory::format_tag::ncdhw; break;
    }

    InferenceEngine::LayerConfig config;
    config.dynBatchSupport = true;
    config.inConfs.resize(getParentEdges().size());
    config.outConfs.resize(1);
    for (size_t i = 0; i < getParentEdges().size(); i++) {
        config.inConfs[i].inPlace = -1;
        config.inConfs[i].constant = i != 0;
        config.inConfs[i].desc = MKLDNNMemoryDesc(getParentEdgeAt(i)->getDims(), inputDataType, i == 0 ? fmt : memory::format_tag::x);
    }
    config.outConfs[0].inPlace = -1;
    config.outConfs[0].constant = false;
    config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);

    supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, fmt);

    if ((inDims.ndims() == 4 || inDims.ndims() == 5) && channelAxis >= 0 && dims[channelAxis] % 8 == 0) {
        fmt = inDims.ndims() == 5 ? memory::format_tag::nCdhw8c : memory::format_tag::nChw8c;
        config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, fmt);
        config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
        supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, fmt);
        if (dims[channelAxis] % 16 == 0) {
            fmt = inDims.ndims() == 5 ? memory::format_tag::nCdhw16c : memory::format_tag::nChw16c;
            config.inConfs[0].desc = MKLDNNMemoryDesc(getParentEdgeAt(0)->getDims(), inputDataType, fmt);
            config.outConfs[0].desc = MKLDNNMemoryDesc(getChildEdgeAt(0)->getDims(), outputDataType, fmt);
            supportedPrimitiveDescriptors.emplace_back(config, impl_desc_type::unknown, fmt);
        }
    }
}

void MKLDNNCropNode::createPrimitive() {
    auto& dstMemPtr = getChildEdgeAt(0)->getMemoryPtr();
    auto& srcMemPtr = getParentEdgeAt(0)->getMemoryPtr();
    if (!dstMemPtr || !dstMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Destination memory didn't allocate.";
    if (!srcMemPtr || !srcMemPtr->GetPrimitivePtr())
        THROW_IE_EXCEPTION << "Input memory didn't allocate.";
    if (getSelectedPrimitiveDescriptor() == nullptr)
        THROW_IE_EXCEPTION << "Preferable primitive descriptor is not set.";
}

void MKLDNNCropNode::execute(mkldnn::stream strm) {
    auto& parentMem = getParentEdgeAt(0)->getMemory();

    int m_block_size = 1;
    if (!parentMem.GetDesc().isPlainFormat()) {
        const auto &desc = parentMem.GetDescriptor().data;
        const auto &blk = desc.format_desc.blocking;
        IE_ASSERT(desc.format_kind == dnnl_blocked &&
                  blk.inner_nblks == 1 &&
                  blk.inner_idxs[0] == 1);
        m_block_size = blk.inner_blks[0];
    }
    const int m_inner_dim = dims[dims.size() - 1] * m_block_size;

    const auto &dst_mem = getChildEdgeAt(0)->getMemory();

    const int dst_ndims = dst_mem.GetDesc().getDims().ndims();

    // TODO: Rewrite it in general case. For every tensor
    // and rank, without using letter N,C,D,H,W
    const int OFFSET_N = (dst_ndims > 0) ? offsets[0] : 0;
    const int OFFSET_C = (dst_ndims > 1) ? offsets[1] : 0;
    const int OFFSET_D = (dst_ndims > 4) ? offsets[offsets.size() - 3] : 0;
    const int OFFSET_H = (dst_ndims > 2) ? offsets[offsets.size() - 2] : 0;
    const int OFFSET_W = (dst_ndims > 3) ? offsets[offsets.size() - 1] : 0;

    // TODO: Check applicability of dyn_batch_lim in early steps.
    //       crop of batch dimension doesn't support dyn batch.
    const int ON = (dst_ndims  > 0) ? std::min<int>(batchToProcess(), getChildEdgeAt(0)->getDims()[0]) : 1;
    const int OC = (dst_ndims  > 1) ? dims[1] : 1;
    const int OD = (dst_ndims  > 4) ? dims[dims.size() - 3] : 1;
    const int OH = (dst_ndims  > 2) ? dims[dims.size() - 2] : 1;
    const int OW = (dst_ndims  > 3) ? dims[dims.size() - 1] : 1;

    memory::dims src_dims = parentMem.GetDims();
    int src_ndims = static_cast<int>(src_dims.size());

    const int IC = (src_ndims  > 1) ? rnd_up(src_dims[1], m_block_size) : 1;
    const int ID = (src_ndims  > 4) ? src_dims[src_dims.size() - 3] : 1;
    const int IH = (src_ndims  > 2) ? src_dims[src_dims.size() - 2] : 1;
    const int IW = (src_ndims  > 3) ? src_dims[src_dims.size() - 1] : 1;

    const size_t itemSize = parentMem.GetDesc().GetElementSize();

    const auto *src_data = reinterpret_cast<const uint8_t*>(parentMem.GetPtr());
    auto *dst_data = reinterpret_cast<uint8_t*>(getChildEdgeAt(0)->getMemory().GetPtr());

    if (OD == 1 && OH == 1 && OW == 1 && ID == 1 && IH == 1 && IW == 1) {
        parallel_for(ON, [&](int n) {
            cpu_memcpy(dst_data + itemSize * n * OC, src_data + itemSize *((n+OFFSET_N)*IC + OFFSET_C), OC * itemSize);
        });
    } else {
        parallel_for2d(ON, (OC / m_block_size), [&](int n, int c) {
            for (int d = 0; d < OD; ++d) {
                int dst_ind = (n*OC + c*m_block_size)*OD*OH*OW + d*m_block_size*OH*OW;

                int src_ind = ((n+OFFSET_N)*IC + (c*m_block_size+OFFSET_C))*ID*IH*IW +
                              ((d+OFFSET_D)*IH*IW + OFFSET_H*IW + OFFSET_W)*m_block_size;

                for (int h = 0; h < OH; ++h) {
                    cpu_memcpy(dst_data + itemSize * dst_ind, src_data + itemSize * src_ind, m_inner_dim * itemSize);

                    src_ind += IW * m_block_size;
                    dst_ind += OW * m_block_size;
                }
            }
        });
    }
}

bool MKLDNNCropNode::created() const {
    return getType() == Crop;
}
REG_MKLDNN_PRIM_FOR(MKLDNNCropNode, Crop);

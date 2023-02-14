// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dnnl_pooling.hpp"
#include "ie_parallel.hpp"
#include <dnnl_extension_utils.h>
#include "onednn/dnnl.h"
#include "common/primitive_cache.hpp"

namespace ov {
namespace intel_cpu {

DnnlPoolingExecutor::DnnlPoolingExecutor(const ExecutorContext::CPtr context) : PoolingExecutor(context) {}

bool DnnlPoolingExecutor::init(const PoolingAttrs& poolingAttrs,
                              const std::vector<MemoryDescPtr>& srcDescs,
                              const std::vector<MemoryDescPtr>& dstDescs,
                              const dnnl::primitive_attr &attr) {
    /*this->stream = dnnl::engine(context->getEngine());
    this->poolingAttrs = poolingAttrs;
    auto localAttrs = dnnl::primitive_attr(attr.get()->clone());
    localAttrs.set_scratchpad_mode(dnnl::scratchpad_mode::user);

    auto desc = createDescriptor(poolingAttrs, srcDescs, dstDescs);
    dnnl::pooling::primitive_desc prim_desc;
    if (!context->getImplPriorities().empty()) {
        for (auto preferredImplType : context->getImplPriorities()) {
            dnnl::primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(context->getEngine(), localAttrs);
            while (static_cast<bool>(itpd))  {
                auto currentImplType = parse_impl_name(itpd.impl_info_str());
                if (currentImplType == preferredImplType) {
                    prim_desc = itpd.get();
                    implType = currentImplType;
                    break;
                }

                if (!itpd.next_impl())
                    break;
            }

            dnnl::pooling::primitive_desc prim_desc = itpd.get();
        }
    } else {
        dnnl::primitive_desc_iterator itpd = desc.createPrimitiveDescriptorIterator(context->getEngine(), localAttrs);
        implType = parse_impl_name(itpd.impl_info_str());
        prim_desc = itpd.get();
    }

    if (!prim_desc)
        return false;

    auto scratchpadMemoryDesc = DnnlExtensionUtils::makeDescriptor(prim_desc.query_md(dnnl::query::scratchpad_md));
    scratchpadMemory = context->getScratchPad()->createScratchPadMem(scratchpadMemoryDesc);

    prim = std::make_shared<dnnl::pooling>(prim_desc);*/

    return true;
}

void DnnlPoolingExecutor::exec(const std::vector<MemoryCPtr>& src, const std::vector<MemoryPtr>& dst, std::unordered_map<int, MemoryPtr> postOpsArgs) {
    /*std::unordered_map<int, dnnl::memory> primArgs;


    primArgs[DNNL_ARG_SCRATCHPAD] = scratchpadMemory->GetPrimitive();
    primArgs[DNNL_ARG_SRC_0] = src[0]->GetPrimitive();
    primArgs[DNNL_ARG_WEIGHTS_0] = src[1]->GetPrimitive();
    primArgs[DNNL_ARG_DST] = dst[0]->GetPrimitive();
    if (poolingAttrs.withBias)
        primArgs[DNNL_ARG_BIAS] = src[2]->GetPrimitive();

    for (auto & entry : postOpsArgs) {
        primArgs[entry.first] = entry.second->GetPrimitive();
    }

    (*prim).execute(stream, primArgs);*/
}

DnnlPoolingExecutor::Key::Key(const PoolingAttrs& poolingAttrs,
                             const std::vector<MemoryDescPtr>& srcDescs,
                             const std::vector<MemoryDescPtr>& dstDescs,
                             const dnnl::primitive_attr &attr) {
    /*this->poolingAttrs = poolingAttrs;
    this->inp0 = MemoryDescUtils::convertToDnnlMemoryDesc(srcDescs[0]);
    this->inp1 = MemoryDescUtils::convertToDnnlMemoryDesc(srcDescs[1]);
    this->bias = poolingAttrs.withBias ? MemoryDescUtils::convertToDnnlMemoryDesc(srcDescs[2]) : nullptr;
    this->out = MemoryDescUtils::convertToDnnlMemoryDesc(dstDescs[0]);
    this->attr = attr;*/
}

size_t DnnlPoolingExecutor::Key::hash() const {
    using namespace dnnl::impl;
    using namespace dnnl::impl::primitive_hashing;

    size_t seed = 0;
    /*seed = hash_combine(seed, poolingAttrs.transposeA);
    seed = hash_combine(seed, poolingAttrs.transposeB);
    for (const auto& ptr : {inp0, inp1, bias, out}) {
        if (ptr) {
            seed = hash_combine(seed, get_md_hash(ptr->getDnnlDesc().data));
        }
    }

    seed = hash_combine(seed, get_attr_hash(*attr.get()));*/
    return seed;
}

bool DnnlPoolingExecutor::Key::operator==(const Key& rhs) const {
    bool retVal = true;
    /*retVal = retVal && poolingAttrs.transposeA == rhs.poolingAttrs.transposeA;
    retVal = retVal && poolingAttrs.transposeB == rhs.poolingAttrs.transposeB;

    if (inp0 != rhs.inp0) {
        retVal = retVal && inp0 && rhs.inp0 && inp0->getDnnlDesc() == rhs.inp0->getDnnlDesc();
    }
    if (inp1 != rhs.inp1) {
        retVal = retVal && inp1 && rhs.inp1 && inp1->getDnnlDesc() == rhs.inp1->getDnnlDesc();
    }
    if (bias != rhs.bias) {
        retVal = retVal && bias && rhs.bias && bias->getDnnlDesc() == rhs.bias->getDnnlDesc();
    }
    if (out != rhs.out) {
        retVal = retVal && out && rhs.out && out->getDnnlDesc() == rhs.out->getDnnlDesc();
    }
    retVal = retVal && *attr.get() == *rhs.attr.get();*/
    return retVal;
}

}   // namespace intel_cpu
}   // namespace ov

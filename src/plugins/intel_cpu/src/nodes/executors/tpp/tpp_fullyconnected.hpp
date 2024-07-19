// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include "cpu_memory.h"
#include "nodes/executors/fullyconnected_config.hpp"

namespace ov {
namespace intel_cpu {

class TPPFCExecutor : public Executor {
public:
    TPPFCExecutor(const FCAttrs& attrs,
                  const PostOps& postOps,
                  const MemoryArgs& memory,
                  const ExecutorContext::CPtr context);

    void execute(const MemoryArgs& memory) override;

    impl_desc_type implType() const override {
        return impl_desc_type::ref;
    }

    // offloads execution data preparation from the exec call
    bool update(const MemoryArgs& memory) override;

    static bool supports(const FCConfig& config);

    void moveMemToNumaNode(int numaNodeID) override;

private:
    const FCAttrs& m_attrs;
    // bool with_bias = false;
};
using TPPFCExecutorPtr = std::shared_ptr<TPPFCExecutor>;

}  // namespace intel_cpu
}  // namespace ov
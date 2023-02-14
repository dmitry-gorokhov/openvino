// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "pooling_list.hpp"

namespace ov {
namespace intel_cpu {

const std::vector<PoolingExecutorDesc>& getPoolingExecutorsList() {
    static std::vector<PoolingExecutorDesc> descs = {
        OV_CPU_INSTANCE_ACL(ExecutorType::Acl, std::make_shared<AclPoolingExecutorBuilder>())
        //OV_CPU_INSTANCE_DNNL(ExecutorType::Dnnl, std::make_shared<DnnlPoolingExecutorBuilder>())
    };

    return descs;
}

}   // namespace intel_cpu
}   // namespace ov
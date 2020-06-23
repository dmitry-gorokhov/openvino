// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_parallel.hpp"

template <typename T0, typename F>
void cpu_parallel_for(const T0& D0, const F& func) {
#if IE_THREAD == IE_THREAD_TBB
    auto work_amount = static_cast<size_t>(D0);
    int nthr = parallel_get_max_threads();
    if (static_cast<size_t>(nthr) > work_amount) nthr = static_cast<int>(work_amount);
    if (nthr == 1) {
        InferenceEngine::for_1d(0, 1, D0, func);
    } else {
        tbb::parallel_for(
                0, nthr,
                [&](int ithr) {
                    InferenceEngine::for_1d(ithr, nthr, D0, func);
                },
                tbb::static_partitioner());
    }
#elif IE_THREAD == IE_THREAD_TBB_AUTO
    const int nthr = parallel_get_max_threads();
    tbb::parallel_for(0, nthr, [&](int ithr) {
        InferenceEngine::for_1d(ithr, nthr, D0, func);
    });
#elif IE_THREAD == IE_THREAD_OMP
#pragma omp parallel
    InferenceEngine::for_1d(parallel_get_thread_num(), parallel_get_num_threads(), D0, func);
#elif IE_THREAD == IE_THREAD_SEQ
    InferenceEngine::for_1d(0, 1, D0, func);
#endif
}

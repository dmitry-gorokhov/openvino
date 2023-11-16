// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Contains logic derived from bitsandbytes
// https://github.com/TimDettmers/bitsandbytes/blob/c82f51c0f784d8a43ebcb9cdefbf94e3f3b9c6c3/csrc/kernels.cu#L223
// implementation.
// Copyright notice from original source file is as follows.

//*******************************************************************************
// Copyright (c) Facebook, Inc. and its affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
//==============================================================================

#include "openvino/core/type/nf4.hpp"

using namespace ov;

float ConvertNF4::dequantize(uint8_t val) {
    static const std::array<float, 16> lookup = {-7.f,
                                                 -6.f,
                                                 -5.f,
                                                 -4.f,
                                                 -3.f,
                                                 -2.f,
                                                 -1.f,
                                                 0.0f,
                                                 1.f,
                                                 2.f,
                                                 3.f,
                                                 4.f,
                                                 5.f,
                                                 6.f,
                                                 7.f,
                                                 8.f};
    return lookup[val];
}

uint8_t ConvertNF4::quantize(float x) {
    if (x > 0.03979014977812767f)
        if (x > 0.3893125355243683f)          // 1
            if (x > 0.6427869200706482f)      // 11
                if (x > 0.8614784181118011f)  // 111
                    return 0b1111;
                else
                    return 0b1110;
            else if (x > 0.5016634166240692f)  // 110
                return 0b1101;
            else
                return 0b1100;
        else if (x > 0.2035212516784668f)  // 10
            if (x > 0.2920137718319893f)   // 101
                return 0b1011;
            else
                return 0b1010;
        else if (x > 0.1202552504837513f)  // 100
            return 0b1001;
        else
            return 0b1000;
    else if (x > -0.33967943489551544f)      // 0
        if (x > -0.13791173323988914f)       // 01
            if (x > -0.045525018125772476f)  // 011
                return 0b0111;
            else
                return 0b0110;
        else if (x > -0.23460740596055984f)  // 010
            return 0b0101;
        else
            return 0b0100;
    else if (x > -0.6106329262256622f)  // 00
        if (x > -0.4599952697753906f)   // 001
            return 0b0011;
        else
            return 0b0010;
    else if (x > -0.8480964004993439f)  // 000
        return 0b0001;
    else
        return 0b0000;
}

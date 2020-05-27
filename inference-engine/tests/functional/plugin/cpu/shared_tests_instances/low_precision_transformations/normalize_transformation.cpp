// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>

#include "low_precision_transformations/normalize_transformation.hpp"
#include "common_test_utils/test_constants.hpp"

using namespace LayerTestsDefinitions;
using namespace InferenceEngine::details;

namespace {
const std::vector<InferenceEngine::Precision> netPrecisions = {
        InferenceEngine::Precision::FP32,
        InferenceEngine::Precision::FP16
};

const std::vector<LayerTransformation::Params> trasformationParamValues = {
    LayerTestsUtils::LayerTransformationParamsFactory::createParamCpu(),
    LayerTestsUtils::LayerTransformationParamsFactory::createParamI8I8(),
    LayerTestsUtils::LayerTransformationParamsFactory::createParamU8I8()
};

const std::vector<bool> fuseMultiply = { true, false };

INSTANTIATE_TEST_CASE_P(LPT, NormalizeTransformation,
    ::testing::Combine(
        ::testing::ValuesIn(netPrecisions),
        ::testing::Values(InferenceEngine::SizeVector({ 1, 96, 32, 32 })),
        ::testing::Values(CommonTestUtils::DEVICE_CPU),
        ::testing::ValuesIn(trasformationParamValues),
        ::testing::ValuesIn(fuseMultiply)),
    NormalizeTransformation::getTestCaseName);
}  // namespace

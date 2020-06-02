// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <unordered_set>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"

#include "ie_util_internal.hpp"
#include "functional_test_utils/low_precision_transformations/layer_transformation.hpp"
#include "low_precision_transformations/convolution.hpp"
#include "low_precision_transformations/scaleshift_to_convolution.hpp"


namespace LayerTestsUtils {

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParamsU8I8() {
    return InferenceEngine::details::LayerTransformation::Params(
        false,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { InferenceEngine::Precision::U8 },
        { InferenceEngine::Precision::I8 });
}

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParamsU8U8() {
    return InferenceEngine::details::LayerTransformation::Params(
        false,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { InferenceEngine::Precision::U8 },
        { InferenceEngine::Precision::U8 });
}

InferenceEngine::details::LayerTransformation::Params LayerTransformationParamsFactory::createParamsI8I8() {
    return InferenceEngine::details::LayerTransformation::Params(
        false,
        true,
        true,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        InferenceEngine::details::LayerTransformation::QuantizedTensorAlignment::None,
        false,
        true,
        true,
        { InferenceEngine::Precision::I8 },
        { InferenceEngine::Precision::I8 });
}

InferenceEngine::details::LowPrecisionTransformer LayerTransformation::getLowPrecisionTransformer(
    const InferenceEngine::details::LayerTransformation::Params& params) const {
    InferenceEngine::details::LowPrecisionTransformer transformer(getLowPrecisionTransformations(params));
    return transformer;
}

InferenceEngine::CNNNetwork LayerTransformation::transform(InferenceEngine::details::LayerTransformation::Params& params) {
    InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImp = cloneNet(InferenceEngine::CNNNetwork(function));

    auto transformer = getLowPrecisionTransformer(params);
    transformer.transform(*cnnNetworkImp);

    return InferenceEngine::CNNNetwork(cnnNetworkImp);
}

InferenceEngine::CNNNetwork LayerTransformation::transform(const InferenceEngine::details::LowPrecisionTransformations& transformations) {
    InferenceEngine::details::CNNNetworkImplPtr cnnNetworkImp = cloneNet(InferenceEngine::CNNNetwork(function));

    InferenceEngine::details::LowPrecisionTransformer transformer(transformations);
    transformer.transform(*cnnNetworkImp);

    return InferenceEngine::CNNNetwork(cnnNetworkImp);
}

void LayerTransformation::checkParentPrecision(const InferenceEngine::CNNLayerPtr& layer, const bool lowPrecision) {
    EXPECT_EQ(1ul, layer->insData.size()) << "insert data count " << layer->insData.size() << " is not correct for layer " << layer->name;
    const InferenceEngine::DataPtr insData = layer->insData[0].lock();
    EXPECT_TRUE(insData != nullptr) << "insert data is nullable";
    const InferenceEngine::Precision precision = insData->getTensorDesc().getPrecision();

    const std::unordered_set<uint8_t> expectedPrecisions = lowPrecision ?
        std::unordered_set<uint8_t>({ InferenceEngine::Precision::U8, InferenceEngine::Precision::I8 }) :
        std::unordered_set<uint8_t>({ InferenceEngine::Precision::FP16, InferenceEngine::Precision::FP32 });
    EXPECT_TRUE((expectedPrecisions.find(precision) != expectedPrecisions.end())) <<
        "actual precision is " << precision;
}

std::string LayerTransformation::toString(const InferenceEngine::details::LayerTransformation::Params& params) {
    std::ostringstream result;
    result <<
        (params.supportAsymmetricQuantization ? "asymmetric" : "symmetric") << "_" <<
        params.precisionsOnActivations << "_" <<
        params.precisionsOnWeights << "_" <<
        params.quantizedTensorAlignmentOnActivations;

    return result.str();
}

}  // namespace LayerTestsUtils

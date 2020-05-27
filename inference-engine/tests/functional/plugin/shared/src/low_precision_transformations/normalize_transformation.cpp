// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/normalize_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"

#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph/pass/visualize_tree.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string NormalizeTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::basicParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    std::tie(netPrecision, inputShapes, targetDevice) = obj.param;

    std::ostringstream result;
    result << "inputShapes=" << CommonTestUtils::vec2str(inputShapes) << "_";
    result << "netPrecision=" << netPrecision.name() << "_";
    result << "targetDevice=" << targetDevice;
    return result.str();
}

void NormalizeTransformation::SetUp() {
    threshold = 10e-5;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    std::tie(netPrecision, inputShape, targetDevice) = this->GetParam();
    auto ngPrc = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngPrc, ngraph::Shape(inputShape));
    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(paramNode->output(0), ngPrc, 256, { 1ul }, { 0.f }, { 10.f }, { 0.f }, { 10.f });

    const auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ 1 }, std::vector<int64_t>{ 1ul });
    const auto normL2 = std::make_shared<ngraph::opset1::NormalizeL2>(fakeQuantize->output(0), axes, 1e-6, ngraph::op::EpsMode::ADD);

    const auto multiplyConst = std::make_shared<ngraph::op::Constant>(ngPrc, ngraph::Shape(inputShape), std::vector<float>{ 2.f });
    const auto multiply = std::make_shared<ngraph::opset1::Multiply>(normL2->output(0), multiplyConst);

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(multiply)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { paramNode }, "NormalizeTransformation");

    // TODO: move to some another place
    validate();
}

void NormalizeTransformation::validate() {
    const InferenceEngine::CNNNetwork network = transform();

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    IE_SUPPRESS_DEPRECATED_START

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = it->second->getCreatorLayer().lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(NormalizeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions

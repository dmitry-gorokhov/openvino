// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include <memory>
#include <string>

#include <openvino/opsets/opset10.hpp>
#include <transformations/common_optimizations/nonzero_fusion.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

enum BranchType {
    NONZERO_I32,
    NONZERO_I64,
    WO_NONZERO
};

struct NonZeroFusionBuilder {
    NonZeroFusionBuilder() = default;
    NonZeroFusionBuilder(const std::vector<BranchType>& props) : branch_props(props) {}
    
    std::shared_ptr<ov::Model> getOriginal() {
        const auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        ov::NodeVector results;
        for (size_t i = 0; i < branch_props.size(); ++i) {
            std::shared_ptr<ov::Node> nonzero;
            switch (branch_props[i]) {
                case NONZERO_I32:
                    nonzero = std::make_shared<ov::opset10::NonZero>(input, ov::element::i32);
                    break;
                case NONZERO_I64:
                    nonzero = std::make_shared<ov::opset10::NonZero>(input, ov::element::i64);
                    break;
                default:
                    nonzero = input;
                    break;
            }
            auto last_node = std::make_shared<ov::opset10::Relu>(nonzero);
            last_node->set_friendly_name("last_node_" + std::to_string(i));
            results.push_back(last_node);
        }
        return std::make_shared<ov::Model>(results, ov::ParameterVector{input});
    };

    std::shared_ptr<ov::Model> getReference() {
        const auto input = std::make_shared<ov::opset10::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));

        std::shared_ptr<ov::Node> i32_node;
        std::shared_ptr<ov::Node> i64_node;
        ov::NodeVector results;
        for (size_t i = 0; i < branch_props.size(); ++i) {
            std::shared_ptr<ov::Node> nonzero;
            if (branch_props[i] == NONZERO_I32) {
                nonzero = i32_node ? i32_node : std::make_shared<ov::opset10::NonZero>(input, ov::element::i32);
                if (!i32_node)
                    i32_node = nonzero;
            } else if (branch_props[i] == NONZERO_I64) {
                nonzero = i64_node ? i64_node : std::make_shared<ov::opset10::NonZero>(input, ov::element::i64);
                if (!i64_node)
                    i64_node = nonzero;
            } else {
                nonzero = input;
            }
            auto last_node = std::make_shared<ov::opset10::Relu>(nonzero);
            last_node->set_friendly_name("last_node_" + std::to_string(i));
            results.push_back(last_node);
        }
        return std::make_shared<ov::Model>(results, ov::ParameterVector{input});
    }

    std::vector<BranchType> branch_props;
};

class NonZeroFusionTests : public testing::WithParamInterface<std::vector<BranchType>>, public TransformationTestsF {
public:
    NonZeroFusionTests() : TransformationTestsF() {
        comparator.enable(FunctionsComparator::CONSUMERS_COUNT);
    }

    static std::string getTestCaseName(testing::TestParamInfo<std::vector<BranchType>> obj) {
        const std::vector<BranchType> testValues = obj.param;
        std::ostringstream result;
        result << "branch_props_{";
        for (const auto& value : testValues) {
            switch (value) {
                case NONZERO_I32:
                    result << "nonzero_i32,";
                    break;
                case NONZERO_I64:
                    result << "nonzero_i64,";
                    break;
                default:
                    result << "wo_nonzero,";
                    break;
            }
        }
        result << "}";
        return result.str();
    }

protected:
    void SetUp() override {
        const auto branch_props = GetParam();
        builder = NonZeroFusionBuilder(branch_props);
        manager.register_pass<ov::pass::NonZeroFusion>();
    }

    NonZeroFusionBuilder builder;
};

TEST_P(NonZeroFusionTests, NonZeroFusion) {
    model = builder.getOriginal();
    model_ref = builder.getReference();
}

namespace NonZeroFusionTestsInstantiation {
std::vector<std::vector<BranchType>> test_params{
    std::vector<BranchType>(5, BranchType::NONZERO_I32),
    std::vector<BranchType>(5, BranchType::NONZERO_I64),
    std::vector<BranchType>(2, BranchType::WO_NONZERO),
    {BranchType::NONZERO_I32, BranchType::NONZERO_I64, BranchType::NONZERO_I32, BranchType::NONZERO_I64, BranchType::NONZERO_I32},
    {BranchType::NONZERO_I32, BranchType::NONZERO_I64, BranchType::WO_NONZERO, BranchType::NONZERO_I64, BranchType::NONZERO_I32},
    {BranchType::WO_NONZERO, BranchType::NONZERO_I64, BranchType::WO_NONZERO, BranchType::NONZERO_I64, BranchType::NONZERO_I32}
};
INSTANTIATE_TEST_SUITE_P(TransformationTestsF, NonZeroFusionTests, ::testing::ValuesIn(test_params), NonZeroFusionTests::getTestCaseName);

} // namespace NonZeroFusionTestsInstantiation


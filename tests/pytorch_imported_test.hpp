#pragma once

#include <RTNeural.h>
#include "load_csv.hpp"

namespace pytorch_imported_test
{

using TestType = double;

template<typename ModelType>
void processModel(ModelType& model, const std::vector<TestType>& xData, std::vector<TestType>& yData)
{
    model.reset();
    for(size_t n = 0; n < xData.size(); ++n)
    {
        TestType input alignas(RTNEURAL_DEFAULT_ALIGNMENT)[] = { xData[n] };
        yData[n] = model.forward(input);
    }
}

using Vec2d = std::vector<std::vector<double>>;

Vec2d transpose(const Vec2d& x)
{
    auto outer_size = x.size();
    auto inner_size = x[0].size();
    Vec2d y(inner_size, std::vector<double>(outer_size, 0.0f));

    for (size_t i = 0; i < outer_size; ++i)
    {
        for (size_t j = 0; j < inner_size; ++j)
            y[j][i] = x[i][j];
    }

    return y;
}

int pytorch_imported_test()
{
    std::cout << "TESTING PYTORCH IMPORTED MODEL..." << std::endl;
    const std::string pytorch_model = "models/pytorch.json";
    const std::string pytorch_tf_model = "models/pytorch_imported.json";
    const std::string pytorch_x = "test_data/pytorch_x.csv";
    const std::string pytorch_y = "test_data/pytorch_y.csv";
    const std::string pytorch_tf_y = "test_data/pytorch_tf_y.csv";
    constexpr double threshold = 1.0e-12;
    size_t n = 0;

    std::ifstream pytorchX(pytorch_x);
    auto xData = load_csv::loadFile<TestType>(pytorchX);

    std::ifstream pytorchY(pytorch_y);
    auto yRefData1 = load_csv::loadFile<TestType>(pytorchY);

    std::ifstream pytorchTfY(pytorch_tf_y);
    auto yRefData2 = load_csv::loadFile<TestType>(pytorchTfY);

    std::vector<TestType> yData1(xData.size(), (TestType)0);
    {
        std::cout << "Loading non-templated model " << pytorch_tf_model << std::endl;
        std::ifstream jsonStream1(pytorch_tf_model, std::ifstream::binary);
        auto model = RTNeural::json_parser::parseJson<TestType>(jsonStream1, true);
        processModel(*model.get(), xData, yData1);
    }

    std::cout << "Testing pytorch vs RTNeural output:" << std::endl;
    size_t nErrs1 = 0;
    TestType max_error1 = (TestType)0;
    for(n = 0; n < xData.size(); ++n)
    {
        auto err1 = std::abs(yData1[n] - yRefData1[n]);
        if(err1 > threshold)
        {
            max_error1 = std::max(err1, max_error1);
            nErrs1++;

            // For debugging purposes
            // std::cout << "ERR: " << err1 << ", idx: " << n << std::endl;
            // std::cout << yData1[n] << std::endl;
            // std::cout << yRefData1[n] << std::endl;
            // break;
        }
    }

    if(nErrs1 > 0)
    {
        std::cout << "FAIL: " << nErrs1 << " errors!" << std::endl;
        std::cout << "Maximum error: " << max_error1 << std::endl;
    }

    std::cout << "Testing tensorflow vs RTNeural output:" << std::endl;
    size_t nErrs2 = 0;
    TestType max_error2 = (TestType)0;
    for(n = 0; n < xData.size(); ++n)
    {
        auto err2 = std::abs(yData1[n] - yRefData2[n]);
        if(err2 > threshold)
        {
            max_error2 = std::max(err2, max_error2);
            nErrs2++;

            // For debugging purposes
            // std::cout << "ERR: " << err2 << ", idx: " << n << std::endl;
            // std::cout << yData1[n] << std::endl;
            // std::cout << yRefData2[n] << std::endl;
            // break;
        }
    }

    if(nErrs2 > 0)
    {
        std::cout << "FAIL: " << nErrs2 << " errors!" << std::endl;
        std::cout << "Maximum error: " << max_error2 << std::endl;
    }

    std::vector<TestType> yData2(xData.size(), (TestType)0);
    {
        std::cout << "Loading weights from " << pytorch_model << std::endl;
         /* Read PyTorch model file generated from Automated-GuitarAmpModelling */
        std::ifstream jsonStream2(pytorch_model, std::ifstream::binary);
        nlohmann::json weights_json;
        jsonStream2 >> weights_json;

        /* Define a static model to match the one expected @TODO: support multiple models */
        std::cout << "Loading templated model" << std::endl;
        RTNeural::ModelT<TestType, 1, 1,
            RTNeural::LSTMLayerT<TestType, 1, 12>,
            RTNeural::DenseT<TestType, 12, 1>
        > modelT;

        /* Load weights manually @TODO: support GRU */
        auto& lstm = modelT.get<0>();
        auto& dense = modelT.get<1>();
        Vec2d lstm_weights_ih = weights_json["/state_dict/rec.weight_ih_l0"_json_pointer];
        lstm.setWVals(transpose(lstm_weights_ih));

        Vec2d lstm_weights_hh = weights_json["/state_dict/rec.weight_hh_l0"_json_pointer];
        lstm.setUVals(transpose(lstm_weights_hh));

        std::vector<double> lstm_bias_ih = weights_json["/state_dict/rec.bias_ih_l0"_json_pointer];
        std::vector<double> lstm_bias_hh = weights_json["/state_dict/rec.bias_hh_l0"_json_pointer];
        for (int i = 0; i < 64; ++i)
            lstm_bias_hh[i] += lstm_bias_ih[i];
        lstm.setBVals(lstm_bias_hh);

        Vec2d dense_weights = weights_json["/state_dict/lin.weight"_json_pointer];
        dense.setWeights(dense_weights);

        std::vector<double> dense_bias = weights_json["/state_dict/lin.bias"_json_pointer];
        dense.setBias(dense_bias.data());

        processModel(modelT, xData, yData2);
    }

    std::cout << "Testing pytorch vs RTNeural output (manual weights):" << std::endl;
    size_t nErrs3 = 0;
    TestType max_error3 = (TestType)0;
    for(n = 0; n < xData.size(); ++n)
    {
        auto err3 = std::abs(yData2[n] - yRefData1[n]);
        if(err3 > threshold)
        {
            max_error3 = std::max(err3, max_error3);
            nErrs3++;

            // For debugging purposes
            // std::cout << "ERR: " << err3 << ", idx: " << n << std::endl;
            // std::cout << yData2[n] << std::endl;
            // std::cout << yRefData1[n] << std::endl;
            // break;
        }
    }

    if(nErrs3 > 0)
    {
        std::cout << "FAIL: " << nErrs3 << " errors!" << std::endl;
        std::cout << "Maximum error: " << max_error3 << std::endl;
    }

    if(nErrs1 > 0 || nErrs2 > 0 || nErrs3 > 0)
        return 1;

    std::cout << "SUCCESS" << std::endl;
    return 0;
}

} // namespace pytorch_imported_test

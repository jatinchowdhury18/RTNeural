#pragma once

#include <RTNeural.h>
#include <cstring>
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

/* TEST DESCRIPTION:
 *
 * Automated-GuitarAmpModelling is using PyTorch to train SimpleRNN models. At the end of
 * training, Automated-GuitarAmpModelling saves the weights into json files (model.json, model_best.json).
 * These files can be used in RTNeural either by copying the weights into the RTNeural model object by hand
 * or by using a python script to convert directly the weights into a json file that is RTNeural-compliant.
 *
 * The test wants to clarify the following points:
 * - difference between PyTorch and RTNeural outputs;
 * - various techniques to import weights;
 */
int pytorch_imported_test()
{
    std::cout << "TESTING MODEL IMPORTED FROM PYTORCH..." << std::endl;
    const std::string pytorch_model = "models/pytorch.json";
    const std::string pytorch_tf_model = "models/pytorch_imported.json";
    const std::string pytorch_x = "test_data/pytorch_x.csv";
    const std::string pytorch_y = "test_data/pytorch_y.csv";
    constexpr double threshold = 1.0e-12;
    size_t n = 0;

    std::string type1;
    std::string type2;
    int hidden_size1;
    int hidden_size2;
    std::string type;
    int hidden_size;
    nlohmann::json modelData;

    std::ifstream pytorchX(pytorch_x);
    auto xData = load_csv::loadFile<TestType>(pytorchX);

    std::ifstream pytorchY(pytorch_y);
    auto yRefData1 = load_csv::loadFile<TestType>(pytorchY);

    std::vector<TestType> yData1(xData.size(), (TestType)0);
    {
        std::cout << "Loading non-templated model " << pytorch_tf_model << std::endl;
        std::ifstream jsonStream(pytorch_tf_model, std::ifstream::binary);
        auto model = RTNeural::json_parser::parseJson<TestType>(jsonStream, true);
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

    std::ifstream jsonStream1(pytorch_tf_model, std::ifstream::binary);
    jsonStream1 >> modelData;
    type1 = modelData["layers"][0]["type"];
    hidden_size1 = modelData["layers"][0]["shape"].back().get<int>();
    std::cout << "Using model type=" << type1 << " hidden_size=" << hidden_size1 << std::endl;

    /* Read PyTorch model file generated from Automated-GuitarAmpModelling */
    std::ifstream jsonStream2(pytorch_model, std::ifstream::binary);
    nlohmann::json weights_json;
    jsonStream2 >> weights_json;

    type2 = weights_json["/model_data/unit_type"_json_pointer];
    for (int i=0; i<type2.size(); i++)
        type2[i] = tolower(type2[i]);
    hidden_size2 = weights_json["/model_data/hidden_size"_json_pointer];

    if((type1 != type2) || (hidden_size1 != hidden_size2)) {
        std::cout << "Sorry, model files don't match" << std::endl;
        return 1;
    }

    if(hidden_size1 != 12) {
        std::cout << "Sorry, only LSTM-12 or GRU-12 models are supported in this tests" << std::endl;
        return 1;
    }

    type = type1;
    hidden_size = hidden_size1;

    /* Weights for recurrent layer */
    Vec2d rec_weights_ih = weights_json["/state_dict/rec.weight_ih_l0"_json_pointer];
    Vec2d rec_weights_hh = weights_json["/state_dict/rec.weight_hh_l0"_json_pointer];

    std::vector<double> rec_bias_ih = weights_json["/state_dict/rec.bias_ih_l0"_json_pointer];
    std::vector<double> rec_bias_hh = weights_json["/state_dict/rec.bias_hh_l0"_json_pointer];

    /* Weights for output dense layer */
    Vec2d dense_weights = weights_json["/state_dict/lin.weight"_json_pointer];
    std::vector<double> dense_bias = weights_json["/state_dict/lin.bias"_json_pointer];

    std::vector<TestType> yData2(xData.size(), (TestType)0);
    {
        std::cout << "Loading weights from " << pytorch_model << std::endl;

        /* Define a static model to match the one expected @TODO: support multiple models */
        std::cout << "Loading templated model" << std::endl;
        if(type == "lstm") {
            std::cout << "Loading LSTM type" << std::endl;
            RTNeural::ModelT<TestType, 1, 1,
                RTNeural::LSTMLayerT<TestType, 1, 12>,
                RTNeural::DenseT<TestType, 12, 1>
            > modelT;

            /* Load weights manually */
            auto& lstm = modelT.get<0>();
            auto& dense = modelT.get<1>();
            lstm.setWVals(transpose(rec_weights_ih));
            lstm.setUVals(transpose(rec_weights_hh));
            for (int i = 0; i < hidden_size*4; ++i)
                rec_bias_hh[i] += rec_bias_ih[i];
            lstm.setBVals(rec_bias_hh);
            dense.setWeights(dense_weights);
            dense.setBias(dense_bias.data());
            processModel(modelT, xData, yData2);
        } else if(type == "gru") {
            std::cout << "Loading GRU type" << std::endl;
            RTNeural::ModelT<TestType, 1, 1,
                RTNeural::GRULayerT<TestType, 1, 12>,
                RTNeural::DenseT<TestType, 12, 1>
            > modelT;

            /* Load weights manually */
            auto& gru = modelT.get<0>();
            auto& dense = modelT.get<1>();
            gru.setWVals(transpose(rec_weights_ih));
            gru.setUVals(transpose(rec_weights_hh));

            std::vector<std::vector<double>> tmp(2, std::vector<double>(hidden_size*3, 0.0));
            for (int i = 0; i < hidden_size*3; ++i)
                tmp[0][i] += rec_bias_ih[i];
            for (int i = 0; i < hidden_size*3; ++i)
                tmp[1][i] += rec_bias_hh[i];
            gru.setBVals(tmp);
            dense.setWeights(dense_weights);
            dense.setBias(dense_bias.data());
            processModel(modelT, xData, yData2);
        }
    }

    std::cout << "Testing pytorch vs RTNeural output (manual weights):" << std::endl;
    size_t nErrs2 = 0;
    TestType max_error2 = (TestType)0;
    for(n = 0; n < xData.size(); ++n)
    {
        auto err2 = std::abs(yData2[n] - yRefData1[n]);
        if(err2 > threshold)
        {
            max_error2 = std::max(err2, max_error2);
            nErrs2++;

            // For debugging purposes
            // std::cout << "ERR: " << err2 << ", idx: " << n << std::endl;
            // std::cout << yData2[n] << std::endl;
            // std::cout << yRefData1[n] << std::endl;
            // break;
        }
    }

    if(nErrs2 > 0)
    {
        std::cout << "FAIL: " << nErrs2 << " errors!" << std::endl;
        std::cout << "Maximum error: " << max_error2 << std::endl;
    }

    if(nErrs1 > 0 || nErrs2 > 0)
        return 1;

    std::cout << "SUCCESS" << std::endl;
    return 0;
}

} // namespace pytorch_imported_test

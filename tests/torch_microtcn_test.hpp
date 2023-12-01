#include "RTNeural/RTNeural.h"

namespace torch_microtcn_test
{
    template <typename T>
    std::vector<std::vector<T>> loadFile2D(std::ifstream& stream)
    {
        std::vector<std::vector<T>> vec;

        std::string line;
        if(stream.is_open()) {
            while(std::getline(stream, line)) {
                std::vector<T> lineVec;
                std::string num;
                for (auto ch : line) {
                    if (ch == ',') {
                        lineVec.push_back(static_cast<T>(std::stod(num)));
                        num.clear();
                        continue;
                    }

                    num.push_back(ch);
                }

                lineVec.push_back(static_cast<T>(std::stod(num)));
                vec.push_back(lineVec);
            }

            stream.close();
        }

        return RTNeural::torch_helpers::detail::transpose(vec);
    }

    int computeCrop(int input_size, int kernel_size, int dilation_rate)
    {
        int output_size = (input_size  - dilation_rate * (kernel_size - 1) - 1) + 1;
        return input_size - output_size;
    }

    template<typename T, int in_ch, int out_ch, int kernel_size, int dilation_rate>
    class TCNBlock 
    {
    public:
        T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_ch];
        RTNeural::Conv1DT<T, in_ch, out_ch, kernel_size, dilation_rate, 1> conv1;
        RTNeural::BatchNorm1DT<T, out_ch> bn;
        RTNeural::PReLUActivationT<T, out_ch> relu;
        RTNeural::Conv1DT<T, in_ch, out_ch, 1, 1, in_ch> res;

        TCNBlock() {
            const auto conv1_weights = std::string { RTNEURAL_ROOT_DIR } + "models/microtcn/conv1.json";
            const auto bn_weights = std::string { RTNEURAL_ROOT_DIR } + "models/microtcn/bn.json";
            const auto relu_weights = std::string { RTNEURAL_ROOT_DIR } + "models/microtcn/relu.json";
            const auto res_weights = std::string { RTNEURAL_ROOT_DIR } + "models/microtcn/res.json";

            std::ifstream conv1_stream (conv1_weights, std::ifstream::binary);
            nlohmann::json conv1_json;
            conv1_stream >> conv1_json;
            RTNeural::torch_helpers::loadConv1D<T>(conv1_json, "", conv1, false);

            // (TODO) purefunctor: add this to torch_helpers?
            std::ifstream bn_stream (bn_weights, std::ifstream::binary);
            nlohmann::json bn_json;
            bn_stream >> bn_json;

            T epsilon = bn_json.at("eps");
            std::vector<T> gamma = bn_json.at("weight");
            std::vector<T> beta = bn_json.at("bias");
            std::vector<T> runningMean = bn_json.at("running_mean");
            std::vector<T> runningVariance = bn_json.at("running_var");

            bn.setEpsilon(epsilon);
            bn.setGamma(gamma);
            bn.setBeta(beta);
            bn.setRunningMean(runningMean);
            bn.setRunningVariance(runningVariance);

            std::ifstream relu_stream (relu_weights, std::ifstream::binary);
            nlohmann::json relu_json;
            relu_stream >> relu_json;
            std::vector<T> alphaVals = relu_json.at("weight");
            relu.setAlphaVals(alphaVals);

            std::ifstream res_stream (res_weights, std::ifstream::binary);
            nlohmann::json res_json;
            res_stream >> res_json;
            RTNeural::torch_helpers::loadConv1D<T>(res_json, "", res, false);
        }

        inline void forward(const T (&ins)[in_ch]) noexcept
        {
            conv1.forward(ins);
            bn.forward(conv1.outs);
            relu.forward(bn.outs);
            res.forward(ins);
            
            for (int i = 0; i < out_ch; ++i)
            {
                outs[i] = relu.outs[i] + res.outs[i];
            }
        }

        void reset() {
            conv1.reset();
            bn.reset();
            relu.reset();
            res.reset();
        }
    };

    template <typename T>
    int testMicroTCN()
    {
        if (std::is_same<T, float>::value)
            std::cout << "TESTING TORCH/CONV1D GROUP MODEL WITH DATA TYPE: FLOAT" << std::endl;
        else
            std::cout << "TESTING TORCH/CONV1D GROUP MODEL WITH DATA TYPE: DOUBLE" << std::endl;

        RTNeural::ModelT<T, 1, 32, TCNBlock<T, 1, 32, 4, 10>> model;
        model.reset();

        std::ifstream modelInputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/microtcn_x.csv" };
        const auto inputs = load_csv::loadFile2d<T, 1>(modelInputsFile);
        std::vector<std::array<T, 32>> outputs {};
        outputs.resize(inputs.size(), {});

        for (size_t i = 0; i < inputs.size(); ++i)
        {
            model.forward(inputs[i].data());
            std::copy(model.getOutputs(), model.getOutputs() + 32, outputs[i].begin());
        }

        std::ifstream modelOutputsFile { std::string { RTNEURAL_ROOT_DIR } + "test_data/microtcn_y.csv" };
        const auto expected_y = loadFile2D<T> (modelOutputsFile);

        int crop = computeCrop(static_cast<int>(inputs.size()), 4, 10);

        size_t nErrs = 0;
        T max_error = (T)0;
        for(size_t n = 0; n < expected_y.size(); ++n)
        {
            for(size_t j = 0; j < outputs[crop + n].size(); ++j)
            {
                auto err = std::abs(outputs[crop + n][j] - expected_y[n][j]);
                if(err > (T)1.0e-6)
                {
                    max_error = std::max(err, max_error);
                    nErrs++;

                    // For debugging purposes
                    std::cout << "ERR: " << err << ", idx: " << n << std::endl;
                    std::cout << "Output: " << outputs[crop + n][j] << std::endl;
                    std::cout << "Expected: " << expected_y[n][j] << std::endl;
                    break;
                }
            }
        }

        if(nErrs > 0)
        {
            std::cout << "FAIL: " << nErrs << " errors!" << std::endl;
            std::cout << "Maximum error: " << max_error << std::endl;
            return 1;
        }

        std::cout << "SUCCESS" << std::endl;
        return 0;
    }
}

int microtcn_test()
{
    int result = 0;
    result |= torch_microtcn_test::testMicroTCN<float>();
    result |= torch_microtcn_test::testMicroTCN<double>();
    return result;
}

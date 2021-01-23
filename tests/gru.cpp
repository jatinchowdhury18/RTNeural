#include <iostream>
#include <fstream>
#include "model_loader.hpp"
#include "load_csv.hpp"

using T = double;

int main()
{
    std::cout << "TESTING GRU IMPLEMENTATION..." << std::endl;

    std::ifstream jsonStream("models/gru.json", std::ifstream::binary);
    auto model = json_parser::parseJson<T>(jsonStream);

    std::ifstream pythonX("test_data/gru_x_python.csv");
    auto xData = load_csv::loadFile<T>(pythonX);

    std::ifstream pythonY("test_data/gru_y_python.csv");
    const auto yRefData = load_csv::loadFile<T>(pythonY);

    std::vector<T> yData (xData.size(), (T) 0);
    for(size_t n = 0; n < xData.size(); ++n)
    {
        T input[] = { xData[n] };
        yData[n] = model->forward(input);
    }

    constexpr T THRESH = 1.0e-7;
    size_t nErrs = 0;
    for(size_t n = 0; n < xData.size(); ++n)
    {
        auto err = std::abs(yData[n] - yRefData[n]);
        if(err > THRESH)
        {
            // For debugging purposes
            // std::cout << "ERR: " << err << ", idx: " << n << std::endl;
            // break;
            nErrs++;
        }
    }

    if (nErrs > 0)
    {
        std::cout << "FAIL: " << nErrs << " errors!" << std::endl;
        return 1;
    }

    std::cout << "SUCCESS" << std::endl;
    return 0;
}

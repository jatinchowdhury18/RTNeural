#pragma once

#include <RTNeural.h>

// The idea here is to purposely feed an incorrect JSON file to the parseJson() method,
// and make sure that the parser handles it gracefully.
// @TODO: add more test cases, and provide better error messages when parsing fails.

int badModelTest()
{
    int result = 1;

    RTNeural::ModelT<float, 1, 1,
        RTNeural::LSTMLayerT<float, 1, 16>,
        RTNeural::DenseT<float, 16, 1>> lstm_16;

    std::ifstream jsonStream1("models/bad_lstm.json", std::ifstream::binary);

    try
    {
        lstm_16.parseJson(jsonStream1);
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
        result = 0;
    }

    return result;
}

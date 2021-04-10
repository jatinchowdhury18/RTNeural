#include <RTNeural.h>

int main (int argc, char* argv[])
{
    RTNeural::ModelT<float,
        RTNeural::Dense<float>,
        RTNeural::TanhActivation<float>,
        RTNeural::Conv1D<float>,
        RTNeural::TanhActivation<float>,
        RTNeural::GRULayer<float>,
        RTNeural::Dense<float>
    > model ({ 1, 8, 8, 4, 4, 8, 1 }, {
        { 1, 8 }, // Dense
        { 8 }, // Tanh
        { 8, 4, 3, 2 }, // Conv1D
        { 4 }, // Tanh
        { 4, 8 }, // GRU
        { 8, 1 } // Dense
    });

    model.reset();
    
    float x[] = { 2.0f, 4.0f };
    model.forward (x);
}

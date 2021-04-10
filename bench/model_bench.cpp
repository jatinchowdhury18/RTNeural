#include <RTNeural.h>

int main (int argc, char* argv[])
{
    RTNeural::ModelT<float,
        RTNeural::Dense<float>,
        RTNeural::Dense<float>> model ({2, 4, 1}, {{ 2, 4 }, {4, 1}});

    model.reset();
    
    float x[] = { 2.0f, 4.0f };
    model.forward (x);
}

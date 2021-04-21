#ifndef LAYER_H_INCLUDED
#define LAYER_H_INCLUDED

#include <stddef.h>
#include <string>

#if USE_ACCELERATE
// Dummy defines to make this include safe for JUCE and other libraries
#define Point CarbonDummyPointName
#define Component CarbonDummyCompName
#include <Accelerate/Accelerate.h>
#undef Point
#undef Component
#endif

namespace RTNeural
{

/** Neural network layer */
template <typename T>
class Layer
{
public:
    Layer(size_t in_size, size_t out_size)
        : in_size(in_size)
        , out_size(out_size)
    {
    }

    virtual ~Layer() { }

    virtual std::string getName() const noexcept { return ""; }

    virtual void reset() { }
    virtual void forward(const T* input, T* out) = 0;

    const size_t in_size;
    const size_t out_size;
};

} // namespace RTNeural

#endif // LAYER_H_INCLUDED

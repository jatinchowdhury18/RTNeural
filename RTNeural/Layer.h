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

/** Virtual base class for a generic neural network layer. */
template <typename T>
class Layer
{
public:
    /** Constructs a layer with given input and output size. */
    Layer(int in_size, int out_size)
        : in_size(in_size)
        , out_size(out_size)
    {
    }

    virtual ~Layer() { }

    /** Returns the name of this layer. */
    virtual std::string getName() const noexcept { return ""; }

    /** Resets the state of this layer. */
    virtual void reset() { }

    /** Implements the forward propagation step for this layer. */
    virtual void forward(const T* input, T* out) = 0;

    const int in_size;
    const int out_size;
};

} // namespace RTNeural

#endif // LAYER_H_INCLUDED

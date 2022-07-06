#pragma once

#include <RTNeural/RTNeural.h>

/** Static implementation of a custom gated activation layer. */
template <typename T, int in_sizet>
class GatedActivation
{
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_in_size = RTNeural::ceil_div(in_sizet, v_size);
    static constexpr auto v_out_size = RTNeural::ceil_div(in_sizet / 2, v_size);

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = in_sizet / 2;
    static_assert(in_size % v_size == 0, "This implementation expects the input size to be a multiple of the SIMD register width.");

    GatedActivation() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "gated_activation"; }

    /**
     * We're going to tell RTNeural to NOT treat this as an activation
     * layer, since RTNeural expects activation layers to have the same
     * inout/output sizes.
     */
    constexpr bool isActivation() const noexcept { return false; }

    void reset() { }

    /** Performs forward propagation for the gated activation. */
    inline void forward(const v_type (&ins)[v_in_size]) noexcept
    {
        for(int i = 0; i < v_out_size; ++i)
        {
            auto tanh = xsimd::tanh(ins[i]);
            auto sigmoid = (T)1.0 / ((T)1.0 + xsimd::exp(-ins[i + v_out_size]));
            outs[i] = tanh * sigmoid;
        }
    }

    v_type outs[v_out_size];
};

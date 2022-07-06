#pragma once

#include <RTNeural/RTNeural.h>

/** Static implementation of a custom gated activation layer. */
template <typename T, int in_sizet>
class GatedActivation
{
    using v_in_type = Eigen::Matrix<T, in_sizet, 1>;
    using v_out_type = Eigen::Matrix<T, in_sizet / 2, 1>;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = in_size / 2;

    GatedActivation()
        : outs(outs_internal)
    {
        outs = v_out_type::Zero();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "gated_activation"; }

    /**
     * We're going to tell RTNeural to NOT treat this as an activation
     * layer, since RTNeural expects activation layers to have the same
     * inout/output sizes.
     */
    constexpr bool isActivation() const noexcept { return false; }

    void reset() { }

    /** Performs forward propagation for gated activation. */
    inline void forward(const v_in_type& ins) noexcept
    {
        auto tanh = ins.template head<out_size>().array().tanh();
        auto sigmoid = (T)1 / (((T)-1 * ins.template tail<out_size>().array()).array().exp() + (T)1);
        outs = tanh * sigmoid;
    }

    Eigen::Map<v_out_type, RTNeural::RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

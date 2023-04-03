#ifndef DENSEEIGEN_H_INCLUDED
#define DENSEEIGEN_H_INCLUDED

#include "../Layer.h"
#include <Eigen/Dense>

namespace RTNeural
{

/**
 * Dynamic implementation of a fully-connected (dense) layer,
 * with no activation.
 */
template <typename T>
class Dense : public Layer<T>
{
public:
    /** Constructs a dense layer for a given input and output size. */
    Dense(int in_size, int out_size)
        : Layer<T>(in_size, out_size)
    {
        weights = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, in_size);
        bias = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);

        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(in_size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(out_size, 1);
    }

    Dense(std::initializer_list<int> sizes)
        : Dense(*sizes.begin(), *(sizes.begin() + 1))
    {
    }

    Dense(const Dense& other)
        : Dense(other.in_size, other.out_size)
    {
    }

    Dense& operator=(const Dense& other)
    {
        return *this = Dense(other);
    }

    virtual ~Dense() = default;

    /** Returns the name of this layer. */
    std::string getName() const noexcept override { return "dense"; }

    /** Performs forward propagation for this layer. */
    inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        outVec.noalias() = weights * inVec + bias;

        std::copy(outVec.data(), outVec.data() + Layer<T>::out_size, out);
    }

    /**
     * Sets the layer weights from a given vector.
     *
     * The dimension of the weights vector must be
     * weights[out_size][in_size]
     */
    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            for(int k = 0; k < Layer<T>::in_size; ++k)
                weights(i, k) = newWeights[i][k];
    }

    /**
     * Sets the layer weights from a given array.
     *
     * The dimension of the weights array must be
     * weights[out_size][in_size]
     */
    void setWeights(T** newWeights)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            for(int k = 0; k < Layer<T>::in_size; ++k)
                weights(i, k) = newWeights[i][k];
    }

    /**
     * Sets the layer bias from a given array of size
     * bias[out_size]
     */
    void setBias(const T* b)
    {
        for(int i = 0; i < Layer<T>::out_size; ++i)
            bias(i, 0) = b[i];
    }

    /** Returns the weights value at the given indices. */
    T getWeight(int i, int k) const noexcept { return weights(i, k); }

    /** Returns the bias value at the given index. */
    T getBias(int i) const noexcept { return bias(i, 0); }

private:
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> weights;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bias;

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

//====================================================
/**
 * Static implementation of a fully-connected (dense) layer,
 * with no activation.
 */
template <typename T, int in_sizet, int out_sizet>
class DenseT
{
    using vec_type = Eigen::Matrix<T, out_sizet, 1>;
    using mat_type = Eigen::Matrix<T, out_sizet, in_sizet>;

public:
    static constexpr auto in_size = in_sizet;
    static constexpr auto out_size = out_sizet;

    DenseT()
        : outs(outs_internal)
    {
        weights = mat_type::Zero();
        bias = vec_type::Zero();
        outs = vec_type::Zero();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "dense"; }

    /** Returns false since dense is not an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    /** Reset is a no-op, since Dense does not have state. */
    void reset() { }

    /** Performs forward propagation for this layer. */
    inline void forward(const Eigen::Matrix<T, in_size, 1>& ins) noexcept
    {
        outs.noalias() = weights * ins + bias;
    }

    /**
     * Sets the layer weights from a given vector.
     *
     * The dimension of the weights vector must be
     * weights[out_size][in_size]
     */
    void setWeights(const std::vector<std::vector<T>>& newWeights)
    {
        for(int i = 0; i < out_size; ++i)
            for(int k = 0; k < in_size; ++k)
                weights(i, k) = newWeights[i][k];
    }

    /**
     * Sets the layer weights from a given vector.
     *
     * The dimension of the weights array must be
     * weights[out_size][in_size]
     */
    void setWeights(T** newWeights)
    {
        for(int i = 0; i < out_size; ++i)
            for(int k = 0; k < in_size; ++k)
                weights(i, k) = newWeights[i][k];
    }

    /**
     * Sets the layer bias from a given array of size
     * bias[out_size]
     */
    void setBias(const T* b)
    {
        for(int i = 0; i < out_size; ++i)
            bias(i, 0) = b[i];
    }

    Eigen::Map<vec_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

    mat_type weights;
    vec_type bias;
};

} // namespace RTNeural

#endif // DENSEEIGEN_H_INCLUDED

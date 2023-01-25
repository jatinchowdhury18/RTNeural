#ifndef ACTIVATIONEIGEN_H_INCLUDED
#define ACTIVATIONEIGEN_H_INCLUDED

#include "../common.h"

namespace RTNeural
{

/** Dynamic implementation of a tanh activation layer. */
template <typename T>
class TanhActivation : public Activation<T>
{
public:
    /** Constructs a tanh activation layer for a given size. */
    explicit TanhActivation(int size)
        : Activation<T>(size, {}, "tanh")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    TanhActivation(std::initializer_list<int> sizes)
        : TanhActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for tanh activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array().tanh();

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of a tanh activation layer. */
template <typename T, int size>
class TanhActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    TanhActivationT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "tanh"; }

    /** Returns true if this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for tanh activation. */
    inline void forward(const v_type& ins) noexcept
    {
        outs = ins.array().tanh();
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};
/** Dynamic implementation of an approximate tanh activation layer. */
template <typename T>
class FastTanh : public Activation<T>
{
public:
    /** Constructs a tanh activation layer for a given size. */
    explicit FastTanh(int size)
        : Activation<T>(size, {}, "tanh")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    FastTanh(std::initializer_list<int> sizes)
        : FastTanh(*sizes.begin())
    {
    }

    /** Performs forward propagation for tanh activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        outVec = fast_tanh<T>(inVec);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of an approximate tanh activation layer. */
template <typename T, int size>
class FastTanhT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    FastTanhT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "tanh"; }

    /** Returns true if this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for tanh activation. */
    inline void forward(const v_type& ins) noexcept
    {
        outs = fast_tanh<T>(ins);
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a ReLU activation layer. */
template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    /** Constructs a ReLU activation layer for a given size. */
    explicit ReLuActivation(int size)
        : Activation<T>(size, {}, "relu")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    ReLuActivation(std::initializer_list<int> sizes)
        : ReLuActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for ReLU activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array().max((T)0);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of a ReLU activation layer. */
template <typename T, int size>
class ReLuActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    ReLuActivationT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "relu"; }

    /** Returns true since this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for ReLU activation. */
    inline void forward(const v_type& ins) noexcept
    {
        outs = ins.array().max((T)0);
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a sigmoid activation layer. */

template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    /** Constructs a sigmoid activation layer for a given size. */
    explicit SigmoidActivation(int size)
        : Activation<T>(size, {}, "sigmoid")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    SigmoidActivation(std::initializer_list<int> sizes)
        : SigmoidActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for sigmoid activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array();
        sigmoid(outVec);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of a sigmoid activation layer. */
template <typename T, int size>
class SigmoidActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SigmoidActivationT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "sigmoid"; }

    /** Returns true since this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for sigmoid activation. */
    inline void forward(const v_type& ins) noexcept
    {
        outs = (T)1 / (((T)-1 * ins.array()).array().exp() + (T)1);
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a softmax activation layer. */
template <typename T>
class SoftmaxActivation : public Activation<T>
{
public:
    /** Constructs a softmax activation layer for a given size. */
    explicit SoftmaxActivation(int size)
        : Activation<T>(size, {}, "softmax")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    SoftmaxActivation(std::initializer_list<int> sizes)
        : SoftmaxActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for softmax activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array();
        softmax(outVec);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of a softmax activation layer. */
template <typename T, int size>
class SoftmaxActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SoftmaxActivationT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "softmax"; }

    /** Returns true since this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for softmax activation. */
    inline void forward(const v_type& ins) noexcept
    {
        outs = ins.array().exp();
        outs = outs / outs.sum();
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a elu activation layer. */
template <typename T>
class ELuActivation : public Activation<T>
{
public:
    /** Constructs a elu activation layer for a given size. */
    explicit ELuActivation(int size)
        : Activation<T>(size, {}, "elu")
        , ones(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Ones(size, 1))
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    ELuActivation(std::initializer_list<int> sizes)
        : ELuActivation(*sizes.begin())
    {
    }

    /** Performs forward propagation for softmax activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);

        outVec = (inVec.array() > (T)0).select(inVec, alpha * (inVec.array().exp() - ones.array()));
        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;

    /** Sets a custom value for the layer's "alpha" parameter. */
    void set_alpha(T newAlpha) { alpha = newAlpha; }

private:
    const Eigen::Matrix<T, Eigen::Dynamic, 1> ones;
    T alpha = (T)1;
};

/** Static implementation of a elu activation layer. */
template <typename T, int size, int AlphaNumerator = 1, int AlphaDenominator = 1>
class ELuActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    ELuActivationT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "elu"; }

    /** Returns true since this layer is an activation layer. */
    constexpr bool isActivation() const noexcept { return true; }

    void reset() { }

    /** Performs forward propagation for elu activation. */
    template <int A_N = AlphaNumerator, int A_D = AlphaDenominator>
    inline typename std::enable_if<A_N == 1 && A_D == 1, void>::type
    forward(const v_type& ins) noexcept
    {
        outs = (ins.array() > (T)0).select(ins, ins.array().exp() - ones.array());
    }

    /** Performs forward propagation for elu activation (with custom alpha parameter). */
    template <int A_N = AlphaNumerator, int A_D = AlphaDenominator>
    inline typename std::enable_if<A_N != 1 || A_D != 1, void>::type
    forward(const v_type& ins) noexcept
    {
        static constexpr T alpha = (T)AlphaNumerator / (T)AlphaDenominator;
        outs = (ins.array() > (T)0).select(ins, alpha * (ins.array().exp() - ones.array()));
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    const v_type ones = v_type::Ones();
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a PReLU activation layer. */
template <typename T>
class PReLUActivation final : public Activation<T>
{
public:
    explicit PReLUActivation(int size)
        : Activation<T>(size, {}, "prelu")
    {
        alpha = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(size, 1);
        inVec = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(size, 1);
    }

    /** Performs forward propagation for prelu activation. */
    inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);

        outVec = (inVec.array() >= (T)0).select(inVec, alpha.cwiseProduct(inVec));
        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    void setAlphaVals(const std::vector<T>& alphaVals)
    {
        if(alphaVals.size() == 1)
        {
            std::fill(alpha.begin(), alpha.end(), alphaVals[0]);
        }
        else
        {
            std::copy(alphaVals.begin(), alphaVals.end(), alpha.begin());
        }
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;

private:
    Eigen::Matrix<T, Eigen::Dynamic, 1> alpha;
};

/** Static implementation of a PReLU activation layer. */
template <typename T, int size>
class PReLUActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    PReLUActivationT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
        alpha = v_type::Zero();
    }

    /** Returns the name of this layer. */
    std::string getName() const noexcept { return "prelu"; }

    /** Returns false since this layer has weights even though it is an activation layer. */
    constexpr bool isActivation() const noexcept { return false; }

    void reset() { }

    /** Performs forward propagation for prelu activation. */
    inline void forward(const v_type& ins) noexcept
    {
        outs = (ins.array() >= (T)0).select(ins, alpha.cwiseProduct(ins));
    }

    void setAlphaVals(const std::vector<T>& alphaVals)
    {
        if(alphaVals.size() == 1)
        {
            std::fill(std::begin(alpha), std::end(alpha), alphaVals[0]);
        }
        else
        {
            std::copy(alphaVals.begin(), alphaVals.end(), std::begin(alpha));
        }
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    const v_type ones = v_type::Ones();
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
    v_type alpha;
};
} // namespace RTNeural

#endif // ACTIVATIONEIGEN_H_INCLUDED

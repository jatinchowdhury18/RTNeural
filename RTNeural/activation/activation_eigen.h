#ifndef ACTIVATIONEIGEN_H_INCLUDED
#define ACTIVATIONEIGEN_H_INCLUDED

#include "../common.h"
#include "../config.h"
#include "../maths/maths_eigen.h"

namespace RTNEURAL_NAMESPACE
{

/** Dynamic implementation of a tanh activation layer. */
template <typename T, typename MathsProvider = DefaultMathsProvider>
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
    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        outVec = MathsProvider::tanh(inVec);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of a tanh activation layer. */
template <typename T, int size, typename MathsProvider = DefaultMathsProvider>
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

    RTNEURAL_REALTIME void reset() { }

    /** Performs forward propagation for tanh activation. */
    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        outs = MathsProvider::tanh(ins);
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
    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
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

    RTNEURAL_REALTIME void reset() { }

    /** Performs forward propagation for ReLU activation. */
    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        outs = ins.array().max((T)0);
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a sigmoid activation layer. */

template <typename T, typename MathsProvider = DefaultMathsProvider>
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
    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        outVec = MathsProvider::sigmoid(inVec);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of a sigmoid activation layer. */
template <typename T, int size, typename MathsProvider = DefaultMathsProvider>
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

    RTNEURAL_REALTIME void reset() { }

    /** Performs forward propagation for sigmoid activation. */
    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        outs = MathsProvider::sigmoid(ins);
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a softmax activation layer. */
template <typename T, typename MathsProvider = DefaultMathsProvider>
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
    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        outVec = MathsProvider::exp(inVec);
        outVec = outVec / outVec.sum();

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of a softmax activation layer. */
template <typename T, int size, typename MathsProvider = DefaultMathsProvider>
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

    RTNEURAL_REALTIME void reset() { }

    /** Performs forward propagation for softmax activation. */
    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        outs = MathsProvider::exp(ins);
        outs = outs / outs.sum();
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a elu activation layer. */
template <typename T, typename MathsProvider = DefaultMathsProvider>
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
    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);

        outVec = (inVec.array() > (T)0).select(inVec, alpha * (MathsProvider::exp(inVec) - ones.array()));
        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;

    /** Sets a custom value for the layer's "alpha" parameter. */
    RTNEURAL_REALTIME void set_alpha(T newAlpha) { alpha = newAlpha; }

private:
    const Eigen::Matrix<T, Eigen::Dynamic, 1> ones;
    T alpha = (T)1;
};

/** Static implementation of a elu activation layer. */
template <typename T, int size, int AlphaNumerator = 1, int AlphaDenominator = 1, typename MathsProvider = DefaultMathsProvider>
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

    RTNEURAL_REALTIME void reset() { }

    /** Performs forward propagation for elu activation. */
    template <int A_N = AlphaNumerator, int A_D = AlphaDenominator>
    RTNEURAL_REALTIME inline typename std::enable_if<A_N == 1 && A_D == 1, void>::type
    forward(const v_type& ins) noexcept
    {
        outs = (ins.array() > (T)0).select(ins, MathsProvider::exp(ins) - ones.array());
    }

    /** Performs forward propagation for elu activation (with custom alpha parameter). */
    template <int A_N = AlphaNumerator, int A_D = AlphaDenominator>
    RTNEURAL_REALTIME inline typename std::enable_if<A_N != 1 || A_D != 1, void>::type
    forward(const v_type& ins) noexcept
    {
        static constexpr T alpha = (T)AlphaNumerator / (T)AlphaDenominator;
        outs = (ins.array() > (T)0).select(ins, alpha * (MathsProvider::exp(ins) - ones.array()));
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
    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);

        outVec = (inVec.array() >= (T)0).select(inVec, alpha.cwiseProduct(inVec));
        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    RTNEURAL_REALTIME void setAlphaVals(const std::vector<T>& alphaVals)
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

    RTNEURAL_REALTIME void reset() { }

    /** Performs forward propagation for prelu activation. */
    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        outs = (ins.array() >= (T)0).select(ins, alpha.cwiseProduct(ins));
    }

    RTNEURAL_REALTIME void setAlphaVals(const std::vector<T>& alphaVals)
    {
        if(alphaVals.size() == 1)
        {
            std::fill(std::begin(alpha), std::end(alpha), alphaVals[0]);
        }
        else
        {
            for(size_t i = 0; i < (size_t)alpha.size(); i += alphaVals.size())
                std::copy(alphaVals.begin(), alphaVals.end(), std::begin(alpha) + i);
        }
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    const v_type ones = v_type::Ones();
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
    v_type alpha;
};

/** Dynamic implementation of an approximated GeLU activation layer. */
template <typename T, typename MathsProvider = DefaultMathsProvider>
class GeLUActivation : public Activation<T>
{
public:
    explicit GeLUActivation(int size)
        : Activation<T>(size, {}, "gelu")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    GeLUActivation(std::initializer_list<int> sizes)
        : GeLUActivation(*sizes.begin())
    {
    }

    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);

        static const T sqrt_2_over_pi = T(0.45015815807);
        outVec = inVec.array().unaryExpr([sqrt_2_over_pi](T x) {
            T xCube = x * x * x;
            return 0.5 * x * (1.0 + MathsProvider::tanh(sqrt_2_over_pi * (x + 0.044715 * xCube)));
        });

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> outVec;
};

/** Static implementation of an approximated GeLU activation layer. */
template <typename T, int size, typename MathsProvider = DefaultMathsProvider>
class GeLUActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    GeLUActivationT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
    }

    std::string getName() const noexcept { return "gelu"; }

    constexpr bool isActivation() const noexcept { return true; }

    RTNEURAL_REALTIME void reset() { }

    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        static const T sqrt_2_over_pi = T(0.45015815807);
        outs = ins.array().unaryExpr([sqrt_2_over_pi](T x) {
            T xCube = x * x * x;
            return 0.5 * x * (1.0 + MathsProvider::tanh(sqrt_2_over_pi * (x + 0.044715 * xCube)));
        });
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a Swish activation layer. */
template <typename T, typename MathsProvider = DefaultMathsProvider>
class SwishActivation : public Activation<T>
{
public:
    /** set optional beta parameter (default is 1) */ 
    explicit SwishActivation(int size, T beta = (T)1)
        : Activation<T>(size, {}, "swish"), beta(beta)
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    SwishActivation(std::initializer_list<int> sizes, T beta = (T)1)
        : SwishActivation(*sizes.begin(), beta)
    {
    }

    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);

        /** apply swish: x * sigmoid(beta * x) */
        outVec = inVec.array() * MathsProvider::sigmoid(beta * inVec.array());
        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;

private:
    T beta;
};

/** Static implementation of a Swish activation layer. */
template <typename T, int size, typename MathsProvider = DefaultMathsProvider>
class SwishActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    /** set optional beta parameter (default is 1) */ 
    SwishActivationT(T beta = (T)1)
        : outs(outs_internal), beta(beta)
    {
        outs = v_type::Zero(); // initialize output vector to zero
    }

    std::string getName() const noexcept { return "swish"; }

    constexpr bool isActivation() const noexcept { return true; }

    RTNEURAL_REALTIME void reset() { }

    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        /** apply swish: x * sigmoid(beta * x) */
        outs = ins.array() * MathsProvider::sigmoid(beta * ins.array());
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
    T beta;
};

/** Dynamic implementation of a Softplus activation layer. */
template <typename T, typename MathsProvider = DefaultMathsProvider>
class SoftplusActivation : public Activation<T>
{
public:
    explicit SoftplusActivation(int size)
        : Activation<T>(size, {}, "softplus")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    SoftplusActivation(std::initializer_list<int> sizes)
        : SoftplusActivation(*sizes.begin())
    {
    }

    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);

        /** Apply Softplus: ln(1 + exp(x)) */
        outVec = (1 + MathsProvider::exp(inVec.array())).log(); //log() awaiting mathsprovider

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of a Softplus activation layer. */
template <typename T, int size, typename MathsProvider = DefaultMathsProvider>
class SoftplusActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SoftplusActivationT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
    }

    std::string getName() const noexcept { return "softplus"; }

    constexpr bool isActivation() const noexcept { return true; }

    RTNEURAL_REALTIME void reset() { }

    /** Forward propagation for Softplus activation. */
    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        /** Apply Softplus: ln(1 + exp(x)) */
        outs = (1 + MathsProvider::exp(ins.array()).log(); //log() waiting for mathsprovider
    }

    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;

private:
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

/** Dynamic implementation of a Mish activation layer. */
template <typename T, typename MathsProvider = DefaultMathsProvider>
class MishActivation : public Activation<T>
{
public:
    /** Constructor for Mish activation layer depending on the softplus activation internally */
    explicit MishActivation(int size)
        : Activation<T>(size, {}, "mish"), softplus(size)
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    MishActivation(std::initializer_list<int> sizes)
        : MishActivation(*sizes.begin())
    {
    }

    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        /** mish = tanh(softplus(x)) */
        Eigen::Matrix<T, Eigen::Dynamic, 1> softplusOut(Layer<T>::in_size);
        softplus.forward(input, softplusOut.data());

        outVec = MathsProvider::tanh(inVec.array() * softplusOut.array());

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

private:
    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
    SoftplusActivation<T, MathsProvider> softplus; // Softplus instance
};

/** Static implementation of a Mish activation layer. */
template <typename T, int size, typename MathsProvider = DefaultMathsProvider>
class MishActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>; 

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    MishActivationT()
        : outs(outs_internal), softplus()
    {
        outs = v_type::Zero();
    }

    std::string getName() const noexcept { return "mish"; }

    constexpr bool isActivation() const noexcept { return true; }

    RTNEURAL_REALTIME void reset() { }

    /** Forward propagation for Mish activation. */
    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        v_type softplusOut = softplus.forward(ins); // Apply Softplus
        outs = ins.array() * MathsProvider::tanh(softplusOut.array()); // Then apply tanh using MathsProvider
    }

private:
    Eigen::Map<v_type, RTNeuralEigenAlignment> outs; 
    SoftplusActivationT<T, size, MathsProvider> softplus; // Softplus instance
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size]; // Internal storage for outputs
};

/** Dynamic implementation of a CDELU activation layer. */
template <typename T, typename MathsProvider = DefaultMathsProvider>
class CDELUActivation : public Activation<T>
{
public:
    /** Constructor for CDELU activation layer with alpha and beta parameters */
    explicit CDELUActivation(int size, T alpha = (T)1, T beta = (T)0.1)
        : Activation<T>(size, {}, "cdelu"), alpha(alpha), beta(beta)
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    /** Performs forward propagation for CDELU activation. */
    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);
        /** cdelu = x for x > 0; alpha * (exp(x) - 1) + beta * x for x <= 0 */
        outVec = (inVec.array() > (T)0).select(inVec, alpha * (MathsProvider::exp(inVec.array()) - (T)1) + beta * inVec.array());

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

private:
    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
    T alpha; // Parameter alpha
    T beta;  // Parameter beta
};

/** Static implementation of a CDELU activation layer. */
template <typename T, int size, typename MathsProvider = DefaultMathsProvider>
class CDELUActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    /** Constructor for static CDELU activation layer with alpha and beta parameters */
    CDELUActivationT(T alpha = (T)1, T beta = (T)0.1)
        : outs(outs_internal), alpha(alpha), beta(beta)
    {
        outs = v_type::Zero();
    }

    std::string getName() const noexcept { return "cdelu"; }

    constexpr bool isActivation() const noexcept { return true; }

    RTNEURAL_REALTIME void reset() { }

    /** Forward propagation for CDELU activation. */
    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        /** cdelu = x for x > 0; alpha * (exp(x) - 1) + beta * x for x <= 0 */
        outs = (ins.array() > (T)0).select(ins, alpha * (MathsProvider::exp(ins.array()) - (T)1) + beta * ins.array());
    }

private:
    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
    T alpha; // Parameter alpha
    T beta;  // Parameter beta
};

/** Dynamic implementation of a SELU activation layer. */
template <typename T, typename MathsProvider = DefaultMathsProvider>
class SELUActivation : public Activation<T>
{
public:
    explicit SELUActivation(int size)
        : Activation<T>(size, {}, "selu")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    RTNEURAL_REALTIME inline void forward(const T* input, T* out) noexcept override
    {
        static const T alpha = T(1.67326324);
        static const T lambda = T(1.05070098);

        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, RTNeuralEigenAlignment>(
            input, Layer<T>::in_size, 1);

        /** selu: lambda * x for x > 0; lambda * alpha * (exp(x) - 1) for x <= 0 */
        outVec = (inVec.array() > (T)0).select(lambda * inVec.array(), lambda * alpha * (MathsProvider::exp(inVec.array()) - (T)1));

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

private:
    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

/** Static implementation of a SELU activation layer. */
template <typename T, int size, typename MathsProvider = DefaultMathsProvider>
class SELUActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;

public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SELUActivationT()
        : outs(outs_internal)
    {
        outs = v_type::Zero();
    }

    RTNEURAL_REALTIME inline void forward(const v_type& ins) noexcept
    {
        static const T alpha = T(1.67326324);
        static const T lambda = T(1.05070098);

        /** selu: lambda * x for x > 0; lambda * alpha * (exp(x) - 1) for x <= 0 */
        outs = (ins.array() > (T)0).select(lambda * ins.array(), lambda * alpha * (MathsProvider::exp(ins.array()) - (T)1));
    }

private:
    Eigen::Map<v_type, RTNeuralEigenAlignment> outs;
    T outs_internal alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];
};

} // namespace RTNEURAL_NAMESPACE

#endif // ACTIVATIONEIGEN_H_INCLUDED

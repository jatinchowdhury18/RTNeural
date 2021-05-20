#ifndef ACTIVATIONEIGEN_H_INCLUDED
#define ACTIVATIONEIGEN_H_INCLUDED

#include "../common.h"

namespace RTNeural
{

template <typename T>
class TanhActivation : public Activation<T>
{
public:
    TanhActivation(size_t size)
        : Activation<T>(size, {}, "tanh")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    TanhActivation(std::initializer_list<size_t> sizes)
        : TanhActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned16>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array().tanh();

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

template <typename T, size_t size>
class TanhActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    TanhActivationT()
    {
        outs = v_type::Zero();
    }

    std::string getName() const noexcept { return "tanh"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const v_type& ins)
    {
        outs = ins.array().tanh();
    }

    v_type outs;
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(size_t size)
        : Activation<T>(size, {}, "relu")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    ReLuActivation(std::initializer_list<size_t> sizes)
        : ReLuActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned16>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array().max((T)0);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

template <typename T, size_t size>
class ReLuActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    ReLuActivationT()
    {
        outs = v_type::Zero();
    }

    std::string getName() const noexcept { return "relu"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const v_type& ins)
    {
        outs = ins.array().max((T)0);
    }

    v_type outs;
};

template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    SigmoidActivation(size_t size)
        : Activation<T>(size, {}, "sigmoid")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    SigmoidActivation(std::initializer_list<size_t> sizes)
        : SigmoidActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned16>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array();
        sigmoid(outVec);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

template <typename T, size_t size>
class SigmoidActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SigmoidActivationT()
    {
        outs = v_type::Zero();
    }

    std::string getName() const noexcept { return "sigmoid"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const v_type& ins)
    {
        outs = (T)1 / (((T)-1 * ins.array()).array().exp() + (T)1);
    }

    v_type outs;
};

template <typename T>
class SoftmaxActivation : public Activation<T>
{
public:
    SoftmaxActivation(size_t size)
        : Activation<T>(size, {}, "softmax")
    {
        inVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
        outVec = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(size, 1);
    }

    SoftmaxActivation(std::initializer_list<size_t> sizes)
        : SoftmaxActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>, Eigen::Aligned16>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array();
        softmax(outVec);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

template <typename T, size_t size>
class SoftmaxActivationT
{
    using v_type = Eigen::Matrix<T, size, 1>;
public:
    static constexpr auto in_size = size;
    static constexpr auto out_size = size;

    SoftmaxActivationT()
    {
        outs = v_type::Zero();
    }

    std::string getName() const noexcept { return "softmax"; }
    constexpr bool isActivation() const noexcept { return true; }
    void reset() { }

    inline void forward(const v_type& ins)
    {
        outs = ins.array().exp();
        outs = outs / outs.sum();
    }

    v_type outs;
};

} // namespace RTNeural

#endif // ACTIVATIONEIGEN_H_INCLUDED

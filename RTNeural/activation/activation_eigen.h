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
        inVec.resize(size, 1);
        outVec.resize(size, 1);
    }

    TanhActivation(std::initializer_list<size_t> sizes)
        : TanhActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array().tanh();

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

template <typename T>
class ReLuActivation : public Activation<T>
{
public:
    ReLuActivation(size_t size)
        : Activation<T>(size, {}, "relu")
    {
        inVec.resize(size, 1);
        outVec.resize(size, 1);
    }

    ReLuActivation(std::initializer_list<size_t> sizes)
        : ReLuActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array().max((T)0);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

template <typename T>
class SigmoidActivation : public Activation<T>
{
public:
    SigmoidActivation(size_t size)
        : Activation<T>(size, {}, "sigmoid")
    {
        inVec.resize(size, 1);
        outVec.resize(size, 1);
    }

    SigmoidActivation(std::initializer_list<size_t> sizes)
        : SigmoidActivation(*sizes.begin())
    {
    }

    inline void forward(const T* input, T* out) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            input, Layer<T>::in_size, 1);
        outVec = inVec.array();
        sigmoid(outVec);

        std::copy(outVec.data(), outVec.data() + Layer<T>::in_size, out);
    }

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

} // namespace RTNeural

#endif // ACTIVATIONEIGEN_H_INCLUDED

#ifndef CONV1DEIGEN_H_INCLUDED
#define CONV1DEIGEN_H_INCLUDED

#include "../Layer.h"
#include <Eigen/Dense>

namespace RTNeural
{

template <typename T>
class Conv1D : public Layer<T>
{
public:
    Conv1D(size_t in_size, size_t out_size, size_t kernel_size, size_t dilation);
    Conv1D(std::initializer_list<size_t> sizes);
    Conv1D(const Conv1D& other);
    Conv1D& operator=(const Conv1D& other);
    virtual ~Conv1D();

    void reset() override;

    virtual inline void forward(const T* input, T* h) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>(
            input, Layer<T>::in_size, 1);

        // insert input into double-buffered state
        state.col(state_ptr) = inVec;
        state.col(state_ptr + state_size) = inVec;

        for(size_t i = 0; i < Layer<T>::out_size; ++i)
            outVec(i, 0) = state.block(0, state_ptr, Layer<T>::in_size, state_size).cwiseProduct(kernelWeights[i]).sum();

        outVec = outVec + bias;
        std::copy(outVec.data(), outVec.data() + Layer<T>::out_size, h);

        state_ptr = (state_ptr == 0 ? state_size - 1 : state_ptr - 1); // iterate state pointer in reverse
    }

    void setWeights(const std::vector<std::vector<std::vector<T>>>& weights);
    void setBias(const std::vector<T>& biasVals);

    size_t getKernelSize() const noexcept { return kernel_size; }
    size_t getDilationRate() const noexcept { return dilation_rate; }

private:
    const size_t dilation_rate;
    const size_t kernel_size;
    const size_t state_size;

    std::vector<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> kernelWeights;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bias;

    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> state;
    size_t state_ptr = 0;

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> outVec;
};

} // RTNeural

#endif // CONV1DEIGEN_H_INCLUDED

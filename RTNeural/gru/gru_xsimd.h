#ifndef GRUXSIMD_H_INCLUDED
#define GRUXSIMD_H_INCLUDED

#include "../Layer.h"
#include "../common.h"
#include <vector>
namespace RTNeural {

template <typename T> class GRULayer : public Layer<T> {
public:
  GRULayer(size_t in_size, size_t out_size);
  virtual ~GRULayer();

  void reset() override { std::fill(ht1, ht1 + Layer<T>::out_size, (T)0); }

  virtual inline void forward(const T *input, T *h) override {
    for (size_t i = 0; i < Layer<T>::out_size; ++i) {
      zVec[i] = vMult(zWeights.W[i], input, prod_in, Layer<T>::in_size) +
                vMult(zWeights.U[i], ht1, prod_out, Layer<T>::out_size);
      rVec[i] = vMult(rWeights.W[i], input, prod_in, Layer<T>::in_size) +
                vMult(rWeights.U[i], ht1, prod_out, Layer<T>::out_size);
    }

    vAdd(zVec, zWeights.b[0], zVec, Layer<T>::out_size);
    vAdd(zVec, zWeights.b[1], zVec, Layer<T>::out_size);
    sigmoid(zVec, zVec, Layer<T>::out_size);

    vAdd(rVec, rWeights.b[0], rVec, Layer<T>::out_size);
    vAdd(rVec, rWeights.b[1], rVec, Layer<T>::out_size);
    sigmoid(rVec, rVec, Layer<T>::out_size);

    for (size_t i = 0; i < Layer<T>::out_size; ++i)
      cVec[i] =
          vMult(cWeights.W[i], input, prod_in, Layer<T>::in_size) +
          rVec[i] * (vMult(cWeights.U[i], ht1, prod_out, Layer<T>::out_size) +
                     cWeights.b[1][i]);
    vAdd(cVec, cWeights.b[0], cVec, Layer<T>::out_size);
    tanh(cVec, cVec, Layer<T>::out_size);

    vSub(ones, zVec, h, Layer<T>::out_size);
    vProd(h, cVec, h, Layer<T>::out_size);
    vProd(zVec, ht1, prod_out, Layer<T>::out_size);
    vAdd(h, prod_out, h, Layer<T>::out_size);

    vCopy(h, ht1, Layer<T>::out_size);
  }

  void setWVals(T **wVals);
  void setUVals(T **uVals);
  void setBVals(T **bVals);

  void setWVals(const std::vector<std::vector<T>> &wVals);
  void setUVals(const std::vector<std::vector<T>> &uVals);
  void setBVals(const std::vector<std::vector<T>> &bVals);

  T getWVal(size_t i, size_t k) const noexcept;
  T getUVal(size_t i, size_t k) const noexcept;
  T getBVal(size_t i, size_t k) const noexcept;

protected:
  T *ht1;

  struct WeightSet {
    WeightSet(size_t in_size, size_t out_size);
    ~WeightSet();

    T **W;
    T **U;
    T *b[2];
    const size_t out_size;
  };

  WeightSet zWeights;
  WeightSet rWeights;
  WeightSet cWeights;

  T *zVec;
  T *rVec;
  T *cVec;

  T *prod_in;
  T *prod_out;
  T *ones;
};

} // namespace RTNeural

#endif // GRUXSIMD_H_INCLUDED

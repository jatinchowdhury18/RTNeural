#ifndef LSTM_EIGEN_INCLUDED
#define LSTM_EIGEN_INCLUDED

#include "../common.h"
#include "../Layer.h"

namespace RTNeural
{

template<typename T>
class LSTMLayer : public Layer<T>
{
public:
    LSTMLayer (size_t in_size, size_t out_size);
    virtual ~LSTMLayer() {}

    void reset() override;
    inline void forward(const T* input, T* h) override
    {
        inVec = Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>> (input, Layer<T>::in_size, 1);

        fVec = Wf * inVec + Uf * ht1 + bf;
        iVec = Wi * inVec + Ui * ht1 + bi;
        oVec = Wo * inVec + Uo * ht1 + bo;
        ctVec = (Wc * inVec + Uc * ht1 + bc).array().tanh();

        sigmoid(fVec);
        sigmoid(iVec);
        sigmoid(oVec);

        cVec = fVec.cwiseProduct(ct1) + iVec.cwiseProduct(ctVec);
        ht1 = cVec.array().tanh();
        ht1 = oVec.cwiseProduct(ht1);

        ct1 = cVec;
        std::copy (ht1.data(), ht1.data() + Layer<T>::out_size, h);
    }

    void setWVals(const std::vector<std::vector<T>>& wVals);
    void setUVals(const std::vector<std::vector<T>>& uVals);
    void setBVals(const std::vector<T>& bVals);

private:
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wf;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wi;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wo;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Wc;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Uf;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Ui;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Uo;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> Uc;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bf;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bi;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bo;
    Eigen::Matrix<T, Eigen::Dynamic, 1> bc;

    Eigen::Matrix<T, Eigen::Dynamic, 1> fVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> iVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> oVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> ctVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> cVec;

    Eigen::Matrix<T, Eigen::Dynamic, 1> inVec;
    Eigen::Matrix<T, Eigen::Dynamic, 1> ht1;
    Eigen::Matrix<T, Eigen::Dynamic, 1> ct1;
};

} // namespace RTNeural

#endif // LSTM_EIGEN_INCLUDED

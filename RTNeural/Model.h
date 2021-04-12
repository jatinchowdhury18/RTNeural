#ifndef MODEL_H_INCLUDED
#define MODEL_H_INCLUDED

#include <iostream>
#include <vector>

#include "Layer.h"
#include "activation/activation.h"
#include "conv1d/conv1d.h"
#include "conv1d/conv1d.tpp"
#include "dense/dense.h"
#include "gru/gru.h"
#include "gru/gru.tpp"
#include "lstm/lstm.h"
#include "lstm/lstm.tpp"

namespace RTNeural
{

/** Neural network model */
template <typename T>
class Model
{
public:
    Model(size_t in_size)
        : in_size(in_size)
    {
    }

    ~Model()
    {
        for(auto l : layers)
            delete l;
        layers.clear();

        outs.clear();
    }

    size_t getNextInSize()
    {
        if(layers.empty())
            return in_size;

        return layers.back()->out_size;
    }

    void addLayer(Layer<T>* layer)
    {
        layers.push_back(layer);
        outs.push_back(vec_type(layer->out_size, (T)0));
    }

    void reset()
    {
        for(auto* l : layers)
            l->reset();
    }

    inline T forward(const T* input)
    {
        layers[0]->forward(input, outs[0].data());

        for(size_t i = 1; i < layers.size(); ++i)
        {
            layers[i]->forward(outs[i - 1].data(), outs[i].data());
        }

        return outs.back()[0];
    }

    inline const T* getOutputs() const noexcept
    {
        return outs.back().data();
    }

    std::vector<Layer<T>*> layers;

private:
#if USE_XSIMD
    using vec_type = std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)>;
#else
    using vec_type = std::vector<T>;
#endif

    const size_t in_size;
    std::vector<vec_type> outs;
};

} // namespace RTNeural

#endif // MODEL_H_INCLUDED

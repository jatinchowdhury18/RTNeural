#pragma once

#include "Model.h"

namespace RTNeural
{

/**
 * Some utilities for constructing and working
 * with variadic templates of layers.
 * 
 * Note that this API may change at any time,
 * so probably don't use any of this directly.
 */
namespace modelt_detail
{

/** Useful for parsing constructor args */
using init_list = std::initializer_list<std::initializer_list<size_t>>;
using Liter = init_list::const_iterator;

/** Forward declaration. */
template <typename... Layers>
struct MakeLayersTupleImpl;

/** Base case. */
template <typename Layer>
struct MakeLayersTupleImpl<Layer>
{
    std::tuple<Layer> operator()(Liter begin, Liter /*end*/) const
    {
        return std::tuple<Layer> (Layer (*begin));
    }
};

/* Recursive case. */
template <typename Layer, typename... Layers>
struct MakeLayersTupleImpl<Layer, Layers...>
{
    std::tuple<Layer, Layers...> operator()(Liter begin, Liter end) const
    {
        return std::tuple_cat(MakeLayersTupleImpl<Layer>()(begin, end),
                              MakeLayersTupleImpl<Layers...>()(begin + 1, end));
    }
};

/* Delegate function. */
template <typename... Layers>
std::tuple<Layers...> makeLayersTuple(init_list l)
{
    return MakeLayersTupleImpl<Layers...>()(l.begin(), l.end());
}

/** utils for making offset index sequences */
template<std::size_t N, typename Seq> struct offset_sequence;

template<std::size_t N, std::size_t... Ints>
struct offset_sequence<N, std::index_sequence<Ints...>>
{
 using type = std::index_sequence<Ints + N...>;
};
template<std::size_t N, typename Seq>
using offset_sequence_t = typename offset_sequence<N, Seq>::type;

/** Functions to do a function for each element in the tuple */
template <typename Fn, typename Tuple, size_t... Ix>
constexpr void forEachInTuple (Fn&& fn, Tuple&& tuple, std::index_sequence<Ix...>) noexcept (noexcept (std::initializer_list<int> { (fn (std::get<Ix> (tuple), Ix), 0)... }))
{
    (void) std::initializer_list<int> { ((void) fn (std::get<Ix> (tuple), Ix), 0)... };
}

template <typename T>
using TupleIndexSequence = std::make_index_sequence<std::tuple_size<std::remove_cv_t<std::remove_reference_t<T>>>::value>;

template <typename Fn, typename Tuple>
constexpr void forEachInTuple (Fn&& fn, Tuple&& tuple) noexcept (noexcept (forEachInTuple (std::forward<Fn> (fn), std::forward<Tuple> (tuple), TupleIndexSequence<Tuple> {})))
{
    forEachInTuple (std::forward<Fn> (fn), std::forward<Tuple> (tuple), TupleIndexSequence<Tuple> {});
}

template<size_t start, size_t num>
using TupleIndexSequenceRange = offset_sequence_t<start, std::make_index_sequence<num>>;

template <size_t start, size_t num, typename Fn, typename Tuple>
constexpr void forEachInTupleRange (Fn&& fn, Tuple&& tuple) noexcept (noexcept (forEachInTuple (std::forward<Fn> (fn), std::forward<Tuple> (tuple), TupleIndexSequenceRange<start, num> {})))
{
    forEachInTuple (std::forward<Fn> (fn), std::forward<Tuple> (tuple), TupleIndexSequenceRange<start, num> {});
}

} // namespace modelt_detail

template <typename T, typename... Layers>
class ModelT
{
public:
    using init_list = std::initializer_list<std::initializer_list<size_t>>;
    ModelT(std::initializer_list<size_t> sizes, init_list layer_inits)
        : in_size(*sizes.begin()),
          layers(modelt_detail::makeLayersTuple<Layers...>(layer_inits))
    {
        std::cout << "Constructing..." << std::endl;
        for (size_t i = 0; i < sizes.size(); ++i)
            outs[i-1] = new T[*(sizes.begin() + i)];
        std::cout << "Done Constructing..." << std::endl;
    }

    ~ModelT()
    {
        for(auto o : outs)
            delete[] o;
    }

    void reset()
    {
        std::cout << "Resetting..." << std::endl;
        modelt_detail::forEachInTuple ([&] (auto& layer, size_t) { layer.reset(); }, layers);
    }

    inline T forward(const T* input)
    {
        std::cout << "Inferencing..." << std::endl;
        std::get<0>(layers).forward(input, outs[0]);

        modelt_detail::forEachInTupleRange<1, n_layers-1> ([&] (auto& layer, size_t i) { 
            layer.forward (outs[i-1], outs[i]);
        }, layers);

        return outs.back()[0];
    }

private:
    const size_t in_size;
    std::tuple<Layers...> layers;
    std::array<T*, sizeof...(Layers)> outs;

    static constexpr size_t n_layers = sizeof...(Layers);
};

} // namespace RTNeural

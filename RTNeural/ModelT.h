#pragma once

#include "model_loader.h"

#if USE_XSIMD // for now only implemented

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
    /** utils for making offset index sequences */
    template <std::size_t N, typename Seq>
    struct offset_sequence;

    template <std::size_t N, std::size_t... Ints>
    struct offset_sequence<N, std::index_sequence<Ints...>>
    {
        using type = std::index_sequence<Ints + N...>;
    };
    template <std::size_t N, typename Seq>
    using offset_sequence_t = typename offset_sequence<N, Seq>::type;

    /** Functions to do a function for each element in the tuple */
    template <typename Fn, typename Tuple, size_t... Ix>
    constexpr void forEachInTuple(Fn&& fn, Tuple&& tuple, std::index_sequence<Ix...>) noexcept(noexcept(std::initializer_list<int> { (fn(std::get<Ix>(tuple), Ix), 0)... }))
    {
        (void)std::initializer_list<int> { ((void)fn(std::get<Ix>(tuple), Ix), 0)... };
    }

    template <typename T>
    using TupleIndexSequence = std::make_index_sequence<std::tuple_size<std::remove_cv_t<std::remove_reference_t<T>>>::value>;

    template <typename Fn, typename Tuple>
    constexpr void forEachInTuple(Fn&& fn, Tuple&& tuple) noexcept(noexcept(forEachInTuple(std::forward<Fn>(fn), std::forward<Tuple>(tuple), TupleIndexSequence<Tuple> {})))
    {
        forEachInTuple(std::forward<Fn>(fn), std::forward<Tuple>(tuple), TupleIndexSequence<Tuple> {});
    }

    template <size_t start, size_t num>
    using TupleIndexSequenceRange = offset_sequence_t<start, std::make_index_sequence<num>>;

    template <size_t start, size_t num, typename Fn, typename Tuple>
    constexpr void forEachInTupleRange(Fn&& fn, Tuple&& tuple) noexcept(noexcept(forEachInTuple(std::forward<Fn>(fn), std::forward<Tuple>(tuple), TupleIndexSequenceRange<start, num> {})))
    {
        forEachInTuple(std::forward<Fn>(fn), std::forward<Tuple>(tuple), TupleIndexSequenceRange<start, num> {});
    }

    // unrolled loop for forward inferencing
    template <size_t idx, size_t Niter>
    struct forward_unroll
    {
        template <typename T>
        static void call(T& t)
        {
            std::get<idx>(t).forward(std::get<idx - 1>(t).outs);
            forward_unroll<idx + 1, Niter - 1>::call(t);
        }
    };

    template <size_t idx>
    struct forward_unroll<idx, 0>
    {
        template <typename T>
        static void call(T&) { }
    };

    template <typename T, typename LayerType>
    void loadLayer(LayerType&, size_t&, const nlohmann::json&, const std::string&, size_t, bool debug)
    {
        json_parser::debug_print("Loading a no-op layer!", debug);
    }

    template <typename T, size_t in_size, size_t out_size>
    void loadLayer(DenseT<T, in_size, out_size>& dense, size_t& json_stream_idx, const nlohmann::json& l,
        const std::string& type, size_t layerDims, bool debug)
    {
        using namespace json_parser;

        debug_print("Layer: " + type, debug);
        debug_print("  Dims: " + std::to_string(layerDims), debug);
        const auto weights = l["weights"];

        if(checkDense<T>(dense, type, layerDims, debug))
            loadDense<T>(dense, weights);

        if(!l.contains("activation"))
        {
            json_stream_idx++;
        }
        else
        {
            const auto activationType = l["activation"].get<std::string>();
            if(activationType.empty())
                json_stream_idx++;
        }
    }

    template <typename T, size_t in_size, size_t out_size>
    void loadLayer(GRULayerT<T, in_size, out_size>& gru, size_t& json_stream_idx, const nlohmann::json& l,
        const std::string& type, size_t layerDims, bool debug)
    {
        using namespace json_parser;

        debug_print("Layer: " + type, debug);
        debug_print("  Dims: " + std::to_string(layerDims), debug);
        const auto weights = l["weights"];

        if(checkGRU<T>(gru, type, layerDims, debug))
            loadGRU<T>(gru, weights);

        json_stream_idx++;
    }

    template <typename T, size_t in_size, size_t out_size>
    void loadLayer(LSTMLayerT<T, in_size, out_size>& lstm, size_t& json_stream_idx, const nlohmann::json& l,
        const std::string& type, size_t layerDims, bool debug)
    {
        using namespace json_parser;

        debug_print("Layer: " + type, debug);
        debug_print("  Dims: " + std::to_string(layerDims), debug);
        const auto weights = l["weights"];

        if(checkLSTM<T>(lstm, type, layerDims, debug))
            loadLSTM<T>(lstm, weights);

        json_stream_idx++;
    }
} // namespace modelt_detail

template <typename T, size_t in_size, size_t out_size, typename... Layers>
class ModelT
{
public:
    ModelT()
    {
#if USE_XSIMD
        for(size_t i = 0; i < v_in_size; ++i)
            v_ins[i] = v_type((T)0);
    }

    /** Get a reference to the layer at index `Index`. */
    template <int Index>
    auto& get() noexcept
    {
        return std::get<Index>(layers);
    }

    /** Get a reference to the layer at index `Index`. */
    template <int Index>
    const auto& get() const noexcept
    {
        return std::get<Index>(layers);
    }

    void reset()
    {
        modelt_detail::forEachInTuple([&](auto& layer, size_t) { layer.reset(); }, layers);
    }

    template <size_t N = in_size>
    inline typename std::enable_if<(N > 1), T>::type
    forward(const T* input)
    {
#if USE_XSIMD
        for(size_t i = 0; i < v_in_size; ++i)
            v_ins[i] = xsimd::load_aligned(input + i * v_size);
#endif
        std::get<0>(layers).forward(v_ins);
        modelt_detail::forward_unroll<1, n_layers - 1>::call(layers);

#if USE_XSIMD
        for(size_t i = 0; i < v_out_size; ++i)
            xsimd::store_aligned(outs + i * v_size, get<n_layers - 1>().outs[i]);
#endif
        return outs[0];
    }

    template <size_t N = in_size>
    inline typename std::enable_if<N == 1, T>::type
    forward(const T* input)
    {
#if USE_XSIMD
        v_ins[0] = (v_type)input[0];
#endif
        std::get<0>(layers).forward(v_ins);
        modelt_detail::forward_unroll<1, n_layers - 1>::call(layers);

#if USE_XSIMD
        for(size_t i = 0; i < v_out_size; ++i)
            xsimd::store_aligned(outs + i * v_size, get<n_layers - 1>().outs[i]);
#endif
        return outs[0];
    }

    inline const T* getOutputs() const noexcept
    {
        return outs.back().data();
    }

    /** Creates a neural network model from a json stream */
    void parseJson(const nlohmann::json& parent, const bool debug = false)
    {
        using namespace json_parser;

        auto shape = parent["in_shape"];
        auto json_layers = parent["layers"];

        if(!shape.is_array() || !json_layers.is_array())
            return;

        const auto nDims = shape.back().get<size_t>();
        debug_print("# dimensions: " + std::to_string(nDims), debug);

        if(nDims != in_size)
        {
            debug_print("Incorrect input size!", debug);
            return;
        }

        size_t json_stream_idx = 0;
        modelt_detail::forEachInTuple([&](auto& layer, size_t) {
            if(json_stream_idx >= json_layers.size())
            {
                debug_print("Too many layers!", debug);
                return;
            }

            const auto l = json_layers.at(json_stream_idx);
            const auto type = l["type"].get<std::string>();
            const auto layerShape = l["shape"];
            const auto layerDims = layerShape.back().get<size_t>();

            if(layer.isActivation()) // activation layers don't need initialisation
            {
                if(!l.contains("activation"))
                {
                    debug_print("No activation layer expected!", debug);
                    return;
                }

                const auto activationType = l["activation"].get<std::string>();
                if(!activationType.empty())
                {
                    debug_print("  activation: " + activationType, debug);
                    checkActivation(layer, activationType, layerDims, debug);
                }

                json_stream_idx++;
                return;
            }

            modelt_detail::loadLayer<T>(layer, json_stream_idx, l, type, layerDims, debug);
        },
            layers);
    }

    /** Creates a neural network model from a json stream */
    void parseJson(std::ifstream& jsonStream, const bool debug = false)
    {
        nlohmann::json parent;
        jsonStream >> parent;
        return parseJson(parent, debug);
    }

private:
#if USE_XSIMD
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = v_type::size;
    static constexpr auto v_in_size = ceil_div(in_size, v_size);
    static constexpr auto v_out_size = ceil_div(out_size, v_size);
#elif USE_EIGEN
    using vec_type = std::vector<T, Eigen::aligned_allocator<T>>;
#else
    using vec_type = std::vector<T>;
#endif

    v_type v_ins[v_in_size];
    T outs alignas(16)[out_size];

    std::tuple<Layers...> layers;
    static constexpr size_t n_layers = sizeof...(Layers);
};

} // namespace RTNeural

#endif

#endif

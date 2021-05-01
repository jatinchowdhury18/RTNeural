#pragma once

#include "model_loader.h"

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
    using init_list = std::vector<std::initializer_list<size_t>>;
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
            return std::tuple<Layer>(Layer(*begin));
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
        template <typename T, typename IO>
        static void call(T& t, IO& io)
        {
            std::get<idx>(t).forward(io[idx - 1].data(), io[idx].data());
            forward_unroll<idx + 1, Niter - 1>::call(t, io);
        }
    };

    template <size_t idx>
    struct forward_unroll<idx, 0>
    {
        template <typename T, typename IO>
        static void call(T&, IO&) { }
    };

} // namespace modelt_detail

template <typename T, typename... Layers>
class ModelT
{
public:
    using init_list = std::vector<std::initializer_list<size_t>>;
    ModelT(std::initializer_list<size_t> sizes, init_list layer_inits)
        : in_size(*sizes.begin())
        , layers(modelt_detail::makeLayersTuple<Layers...>(layer_inits))
        , sizes(sizes)
        , layer_inits(layer_inits)
    {
        for(size_t i = 1; i < sizes.size(); ++i)
        {
            auto out_size = *(sizes.begin() + i);
            outs[i - 1].resize(out_size, (T)0);
        }
    }

    ModelT(const ModelT& other)
        : ModelT(other.sizes, other.layer_inits)
    {
    }

    ModelT& operator=(const ModelT& other)
    {
        return *this = ModelT(other);
    }

    ~ModelT()
    {
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

    inline T forward(const T* input)
    {
        std::get<0>(layers).forward(input, outs[0].data());
        modelt_detail::forward_unroll<1, n_layers - 1>::call(layers, outs);

        return outs.back()[0];
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

            if(auto* actLayer = dynamic_cast<Activation<T>*>(&layer)) // activation layers don't need initialisation
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
                    checkActivation(*actLayer, activationType, layerDims, debug);
                }

                json_stream_idx++;
                return;
            }

            debug_print("Layer: " + type, debug);
            debug_print("  Dims: " + std::to_string(layerDims), debug);
            const auto weights = l["weights"];

            if(auto* dense = dynamic_cast<Dense<T>*>(&layer))
            {
                if(checkDense(*dense, type, layerDims, debug))
                    loadDense(*dense, weights);

                if(!l.contains("activation"))
                    json_stream_idx++;
            }
            else if(auto* conv = dynamic_cast<Conv1D<T>*>(&layer))
            {
                const auto kernel_size = l["kernel_size"].back().get<size_t>();
                const auto dilation = l["dilation"].back().get<size_t>();

                if(checkConv1D(*conv, type, layerDims, kernel_size, dilation, debug))
                    loadConv1D(*conv, kernel_size, dilation, weights);

                if(!l.contains("activation"))
                    json_stream_idx++;
            }
            else if(auto* gru = dynamic_cast<GRULayer<T>*>(&layer))
            {
                if(checkGRU(*gru, type, layerDims, debug))
                    loadGRU(*gru, weights);

                json_stream_idx++;
            }
            else if(auto* lstm = dynamic_cast<LSTMLayer<T>*>(&layer))
            {
                if(checkLSTM(*lstm, type, layerDims, debug))
                    loadLSTM(*lstm, weights);

                json_stream_idx++;
            }
            else
            {
                debug_print("Layer type not recognized!", debug);

                if(!l.contains("activation"))
                    json_stream_idx++;
            }
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
    using vec_type = std::vector<T, XSIMD_DEFAULT_ALLOCATOR(T)>;
#elif USE_EIGEN
    using vec_type = std::vector<T, Eigen::aligned_allocator<T>>;
#else
    using vec_type = std::vector<T>;
#endif

    const size_t in_size;
    std::tuple<Layers...> layers;

    static constexpr size_t n_layers = sizeof...(Layers);
    std::array<vec_type, n_layers> outs;

    // needed for copy constructor
    std::initializer_list<size_t> sizes;
    init_list layer_inits;
};

} // namespace RTNeural

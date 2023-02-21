#pragma once

#include "model_loader.h"

#define MODELT_AVAILABLE (!RTNEURAL_USE_ACCELERATE)

#if MODELT_AVAILABLE

namespace RTNeural
{

#ifndef DOXYGEN
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
    void loadLayer(LayerType&, int&, const nlohmann::json&, const std::string&, int, bool debug)
    {
        json_parser::debug_print("Loading a no-op layer!", debug);
    }

    template <typename T, int in_size, int out_size>
    void loadLayer(DenseT<T, in_size, out_size>& dense, int& json_stream_idx, const nlohmann::json& l,
        const std::string& type, int layerDims, bool debug)
    {
        using namespace json_parser;

        debug_print("Layer: " + type, debug);
        debug_print("  Dims: " + std::to_string(layerDims), debug);
        const auto& weights = l["weights"];

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

    template <typename T, int in_size, int out_size, int kernel_size, int dilation_rate, bool dynamic_state>
    void loadLayer(Conv1DT<T, in_size, out_size, kernel_size, dilation_rate, dynamic_state>& conv, int& json_stream_idx, const nlohmann::json& l,
        const std::string& type, int layerDims, bool debug)
    {
        using namespace json_parser;

        debug_print("Layer: " + type, debug);
        debug_print("  Dims: " + std::to_string(layerDims), debug);
        const auto& weights = l["weights"];
        const auto kernel = l["kernel_size"].back().get<int>();
        const auto dilation = l["dilation"].back().get<int>();

        if(checkConv1D<T>(conv, type, layerDims, kernel, dilation, debug))
            loadConv1D<T>(conv, kernel, dilation, weights);

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

    template <typename T, int in_size, int out_size, SampleRateCorrectionMode mode>
    void loadLayer(GRULayerT<T, in_size, out_size, mode>& gru, int& json_stream_idx, const nlohmann::json& l,
        const std::string& type, int layerDims, bool debug)
    {
        using namespace json_parser;

        debug_print("Layer: " + type, debug);
        debug_print("  Dims: " + std::to_string(layerDims), debug);
        const auto& weights = l["weights"];

        if(checkGRU<T>(gru, type, layerDims, debug))
            loadGRU<T>(gru, weights);

        json_stream_idx++;
    }

    template <typename T, int in_size, int out_size, SampleRateCorrectionMode mode>
    void loadLayer(LSTMLayerT<T, in_size, out_size, mode>& lstm, int& json_stream_idx, const nlohmann::json& l,
        const std::string& type, int layerDims, bool debug)
    {
        using namespace json_parser;

        debug_print("Layer: " + type, debug);
        debug_print("  Dims: " + std::to_string(layerDims), debug);
        const auto& weights = l["weights"];

        if(checkLSTM<T>(lstm, type, layerDims, debug))
            loadLSTM<T>(lstm, weights);

        json_stream_idx++;
    }

    template <typename T, int size>
    void loadLayer(PReLUActivationT<T, size>& prelu, int& json_stream_idx, const nlohmann::json& l,
        const std::string& type, int layerDims, bool debug)
    {
        using namespace json_parser;

        debug_print("Layer: " + type, debug);
        debug_print("  Dims: " + std::to_string(layerDims), debug);
        const auto& weights = l["weights"];

        if(checkPReLU<T>(prelu, type, layerDims, debug))
            loadPReLU<T>(prelu, weights);

        json_stream_idx++;
    }

    template <typename T, int size, bool affine>
    void loadLayer(BatchNorm1DT<T, size, affine>& batch_norm, int& json_stream_idx, const nlohmann::json& l,
        const std::string& type, int layerDims, bool debug)
    {
        using namespace json_parser;

        debug_print("Layer: " + type, debug);
        debug_print("  Dims: " + std::to_string(layerDims), debug);
        const auto& weights = l["weights"];

        if(checkBatchNorm<T>(batch_norm, type, layerDims, weights, debug))
        {
            loadBatchNorm<T>(batch_norm, weights);
            batch_norm.setEpsilon(l["epsilon"].get<float>());
        }

        json_stream_idx++;
    }
} // namespace modelt_detail
#endif // DOXYGEN

/**
 *  A static sequential neural network model.
 *
 *  To use this class, you must define the layers at compile-time:
 *  ```
 *  ModelT<double, 1, 1,
 *      DenseT<double, 1, 8>,
 *      TanhActivationT<double, 8>,
 *      DenseT<double, 8, 1>
 *  > model;
 *  ```
 */
template <typename T, int in_size, int out_size, typename... Layers>
class ModelT
{
public:
    static constexpr auto input_size = in_size;
    static constexpr auto output_size = out_size;

    ModelT()
    {
#if RTNEURAL_USE_XSIMD
        for(int i = 0; i < v_in_size; ++i)
            v_ins[i] = v_type((T)0);
#elif RTNEURAL_USE_EIGEN
        auto& layer_outs = get<n_layers - 1>().outs;
        new(&layer_outs) Eigen::Map<Eigen::Matrix<T, out_size, 1>, RTNeuralEigenAlignment>(outs);
#endif
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

    /** Resets the state of the network layers. */
    void reset()
    {
        modelt_detail::forEachInTuple([&](auto& layer, size_t)
            { layer.reset(); },
            layers);
    }

    /** Performs forward propagation for this model. */
    template <int N = in_size>
    inline typename std::enable_if<(N > 1), T>::type
    forward(const T* input)
    {
#if RTNEURAL_USE_XSIMD
        for(int i = 0; i < v_in_size; ++i)
            v_ins[i] = xsimd::load_aligned(input + i * v_size);
#elif RTNEURAL_USE_EIGEN
        auto v_ins = Eigen::Map<const vec_type, RTNeuralEigenAlignment>(input);
#else // RTNEURAL_USE_STL
        std::copy(input, input + in_size, v_ins);
#endif
        std::get<0>(layers).forward(v_ins);
        modelt_detail::forward_unroll<1, n_layers - 1>::call(layers);

#if RTNEURAL_USE_XSIMD
        for(int i = 0; i < v_out_size; ++i)
            xsimd::store_aligned(outs + i * v_size, get<n_layers - 1>().outs[i]);
#elif RTNEURAL_USE_EIGEN
#else // RTNEURAL_USE_STL
        auto& layer_outs = get<n_layers - 1>().outs;
        std::copy(layer_outs, layer_outs + out_size, outs);
#endif
        return outs[0];
    }

    /** Performs forward propagation for this model. */
    template <int N = in_size>
    inline typename std::enable_if<N == 1, T>::type
    forward(const T* input)
    {
#if RTNEURAL_USE_XSIMD
        v_ins[0] = (v_type)input[0];
#elif RTNEURAL_USE_EIGEN
        const auto v_ins = vec_type::Constant(input[0]);
#else // RTNEURAL_USE_STL
        v_ins[0] = input[0];
#endif

        std::get<0>(layers).forward(v_ins);
        modelt_detail::forward_unroll<1, n_layers - 1>::call(layers);

#if RTNEURAL_USE_XSIMD
        for(int i = 0; i < v_out_size; ++i)
            xsimd::store_aligned(outs + i * v_size, get<n_layers - 1>().outs[i]);
#elif RTNEURAL_USE_EIGEN
#else // RTNEURAL_USE_STL
        auto& layer_outs = get<n_layers - 1>().outs;
        std::copy(layer_outs, layer_outs + out_size, outs);
#endif
        return outs[0];
    }

    /** Returns a pointer to the output of the final layer in the network. */
    inline const T* getOutputs() const noexcept
    {
        return outs;
    }

    /** Loads neural network model weights from a json stream. */
    void parseJson(const nlohmann::json& parent, const bool debug = false, std::initializer_list<std::string> custom_layers = {})
    {
        using namespace json_parser;

        auto shape = parent["in_shape"];
        auto json_layers = parent["layers"];

        if(!shape.is_array() || !json_layers.is_array())
            return;

        const auto nDims = shape.back().get<int>();
        debug_print("# dimensions: " + std::to_string(nDims), debug);

        if(nDims != in_size)
        {
            debug_print("Incorrect input size!", debug);
            return;
        }

        int json_stream_idx = 0;
        modelt_detail::forEachInTuple([&](auto& layer, size_t)
            {
            if(json_stream_idx >= (int)json_layers.size())
            {
                debug_print("Too many layers!", debug);
                return;
            }

            const auto l = json_layers.at(json_stream_idx);
            const auto type = l["type"].get<std::string>();
            const auto layerShape = l["shape"];
            const auto layerDims = layerShape.back().get<int>();

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

            if(std::find(custom_layers.begin(), custom_layers.end(), type) != custom_layers.end())
            {
                std::cout << "Skipping loading weights for custom layer: " << type << std::endl;
                json_stream_idx++;
                return;
            }

            modelt_detail::loadLayer<T>(layer, json_stream_idx, l, type, layerDims, debug); },
            layers);
    }

    /** Loads neural network model weights from a json stream. */
    void parseJson(std::ifstream& jsonStream, const bool debug = false, std::initializer_list<std::string> custom_layers = {})
    {
        nlohmann::json parent;
        jsonStream >> parent;
        return parseJson(parent, debug, custom_layers);
    }

private:
#if RTNEURAL_USE_XSIMD
    using v_type = xsimd::simd_type<T>;
    static constexpr auto v_size = (int)v_type::size;
    static constexpr auto v_in_size = ceil_div(in_size, v_size);
    static constexpr auto v_out_size = ceil_div(out_size, v_size);
    v_type v_ins[v_in_size];
#elif RTNEURAL_USE_EIGEN
    using vec_type = Eigen::Matrix<T, in_size, 1>;
#else // RTNEURAL_USE_STL
    T v_ins alignas(RTNEURAL_DEFAULT_ALIGNMENT)[in_size];
#endif

    T outs alignas(RTNEURAL_DEFAULT_ALIGNMENT)[out_size];

    std::tuple<Layers...> layers;
    static constexpr size_t n_layers = sizeof...(Layers);
};

} // namespace RTNeural

#endif // MODELT_AVAILABLE

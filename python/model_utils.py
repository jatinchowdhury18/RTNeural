import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
from json import JSONEncoder

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def save_model_json(model, layers_to_skip=(keras.layers.InputLayer)):
    def get_layer_type(layer):
        if isinstance(layer, keras.layers.TimeDistributed):
            return 'time-distributed-dense'

        if isinstance(layer, keras.layers.GRU):
            return 'gru'

        if isinstance(layer, keras.layers.LSTM):
            return 'lstm'

        if isinstance(layer, keras.layers.Dense):
            return 'dense'

        if isinstance(layer, keras.layers.Conv1D):
            return 'conv1d'

        if isinstance(layer, keras.layers.PReLU):
            return 'prelu'

        if isinstance(layer, keras.layers.BatchNormalization):
            if len(layer.input_shape) == 3:
                return 'batchnorm'
            elif len(layer.input_shape) == 4:
                return 'batchnorm2d'
            else :
                raise ValueError("Incorrect input_shape when saving batchnorm layer")
        
        if isinstance(layer, keras.layers.Conv2D):
            return 'conv2d'

        if isinstance(layer, keras.layers.Activation):
            return 'activation'

        return 'unknown'

    def get_layer_activation(layer):
        if isinstance(layer, keras.layers.TimeDistributed):
            return get_layer_activation(layer.layer)

        if not hasattr(layer, 'activation'):
            return ''

        if layer.activation == keras.activations.tanh:
            return 'tanh'

        if layer.activation == keras.activations.relu:
            return 'relu'

        if layer.activation == keras.activations.sigmoid:
            return 'sigmoid'

        if layer.activation == keras.activations.softmax:
            return 'softmax'

        if layer.activation == keras.activations.elu:
            return 'elu'
        
        return ''


    def save_layer(layer):
        layer_dict = {
            "type"       : get_layer_type(layer),
            "activation" : get_layer_activation(layer),
            "shape"      : layer.output_shape,
        }

        if layer_dict["type"] == "conv1d":
            layer_dict["kernel_size"] = layer.kernel_size
            layer_dict["dilation"] = layer.dilation_rate

        if layer_dict["type"] == "conv2d":
            layer_dict["kernel_size_time"] = layer.kernel_size[0]
            layer_dict["kernel_size_feature"] = layer.kernel_size[1]
            layer_dict["dilation"] = layer.dilation_rate[0] # only time axis
            layer_dict["strides"] = layer.strides[1] # only feature axis
            layer_dict["num_filters_in"] = layer.input_shape[3]
            layer_dict["num_features_in"] = layer.input_shape[2]
            layer_dict["num_filters_out"] = layer.output_shape[3]
            layer_dict["padding"] = str(layer.padding).lower()

        if layer_dict["type"] == "batchnorm":
            layer_dict["epsilon"] = layer.epsilon

        if layer_dict["type"] == "batchnorm2d":
            layer_dict["epsilon"] = layer.epsilon
            layer_dict["num_filters_in"] = layer.input_shape[3]
            layer_dict["num_features_in"] = layer.input_shape[2]

        layer_dict["weights"] = layer.get_weights()

        return layer_dict


    model_dict = {}
    model_dict["in_shape"] = model.input_shape
    layers = []
    for layer in model.layers:
        if isinstance(layer, layers_to_skip):
            print(f'Skipping layer: {layer}')
            continue

        layer_dict = save_layer(layer)
        layers.append(layer_dict)

    model_dict["layers"] = layers
    return model_dict

def save_model(model, filename, layers_to_skip=(keras.layers.InputLayer)):
    model_dict = save_model_json(model, layers_to_skip)
    with open(filename, 'w') as outfile:
        json.dump(model_dict, outfile, cls=NumpyArrayEncoder, indent=4)

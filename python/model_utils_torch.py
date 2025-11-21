import numpy as np
import torch
import torch.nn as nn
import json
from itertools import chain

# Swapping functions for GRU parameters
def swap_zr_biases(biases, hidden_size):
    """Swap the 'z' and 'r' gate biases in the biases vector."""
    biases = biases.copy()
    biases[:2*hidden_size] = np.concatenate((
        biases[hidden_size:2*hidden_size],
        biases[:hidden_size]
    ))
    return biases

def swap_zr_weights(weights, hidden_size):
    """Swap the 'z' and 'r' gate weights in the weights matrix."""
    weights = weights.copy()
    weights[:2*hidden_size, :] = np.concatenate((
        weights[hidden_size:2*hidden_size, :],
        weights[:hidden_size, :]
    ), axis=0)
    return weights

# Function to retrieve weights from a layer
def get_weights(layer):
    # LSTM layer
    if isinstance(layer, nn.LSTM):
        # kernel weights: flatten the weight matrix for input-to-hidden connections
        kernel_weights = [layer.weight_ih_l0.view(-1).detach().cpu().numpy().tolist()]
        # recurrent weights: list the columns for each unit
        recurrent_weights = layer.weight_hh_l0.detach().cpu().numpy()
        recurrent_weights = [recurrent_weights[:, i].tolist() for i in range(recurrent_weights.shape[1])]
        # biases (only input biases, since TensorFlow may not use the hidden biases)
        biases = layer.bias_ih_l0.detach().cpu().numpy().tolist()
        return [kernel_weights, recurrent_weights, biases]

    # GRU layer with swapped 'z' and 'r' gate ordering
    elif isinstance(layer, nn.GRU):
        hidden_size = layer.hidden_size

        # Extract and swap the input (kernel) weights
        kernel_weights = layer.weight_ih_l0.detach().cpu().numpy()
        kernel_weights = swap_zr_weights(kernel_weights, hidden_size)
        kernel_weights = [kernel_weights[:, i].tolist() for i in range(kernel_weights.shape[1])]

        # Extract and swap the recurrent weights
        recurrent_weights = layer.weight_hh_l0.detach().cpu().numpy()
        recurrent_weights = swap_zr_weights(recurrent_weights, hidden_size)
        recurrent_weights = [recurrent_weights[:, i].tolist() for i in range(recurrent_weights.shape[1])]

        # Extract and swap biases for input and hidden components
        input_biases = layer.bias_ih_l0.detach().cpu().numpy()
        input_biases = swap_zr_biases(input_biases, hidden_size)
        input_biases = input_biases.tolist()

        hidden_biases = layer.bias_hh_l0.detach().cpu().numpy()
        hidden_biases = swap_zr_biases(hidden_biases, hidden_size)
        hidden_biases = hidden_biases.tolist()

        return [kernel_weights, recurrent_weights, [input_biases, hidden_biases]]

    # Conv1D layer
    elif isinstance(layer, nn.Conv1d):
        # Get the weights and reshape
        weights = layer.weight.detach().cpu().numpy()  # Shape: (out_channels, in_channels, kernel_size)

        # Reverse the rows in the innermost dimension
        reversed_weights = weights[:, :, ::-1]  # This reverses the rows in each kernel

        # Flatten into (out_channels, in_channels * kernel_size)
        flattened_weights = reversed_weights.reshape(reversed_weights.shape[0], -1)

        # Now reshape into the desired structure:

        # Ensure reshaping aligns with dimensions (3, 16, 32) by adjusting shape
        reshaped_weights = flattened_weights.reshape(weights.shape[2],
                                                    weights.shape[1], weights.shape[0])

        # Convert to a list of lists for compatibility
        reshaped_weights = reshaped_weights.tolist()

        # Extract biases
        biases = layer.bias.detach().cpu().numpy().tolist()

        return [reshaped_weights, biases]

    # Dense layer
    elif isinstance(layer, nn.Linear):
      # Modify weight format to match the desired structure
      # Flatten without concatenation
      weights = layer.weight.view(-1).detach().cpu().numpy().tolist()

      # Reformat weights: each weight should be wrapped in a single list
      formatted_weights = [[w] for w in weights]

      # return correctly formatted dense layer weights
      return [formatted_weights,  # Now weights are in the correct format
              layer.bias.detach().cpu().numpy().tolist()]

    # PReLU layer
    elif isinstance(layer, nn.PReLU):
      # PReLU layer has learnable parameters (weights)
      weights = layer.weight.detach().cpu().numpy().tolist()

      # Wrap each weight in a nested list, matching desired JSON structure
      formatted_weights = [[w for w in weights]]
      return [formatted_weights]

    # BatchNorm layers (handle both 1d and 2d)
    elif isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
      # Return the batchnorm weights, biases, and running statistics in the requested format
      weights = [
                  layer.weight.detach().cpu().numpy().tolist(),
                  layer.bias.detach().cpu().numpy().tolist(),
                  layer.running_mean.detach().cpu().numpy().tolist(),
                  layer.running_var.detach().cpu().numpy().tolist()
                ]
      return weights

    # Conv2D layer
    elif isinstance(layer, nn.Conv2d):
      # Get the weights and reshape
      weights = layer.weight.detach().cpu().numpy()

      # Assuming the shape of weights is (out_channels, in_channels, height, width), reshape accordingly
      # First, reshape the weights so each filter's weights are flattened into a vector
      reshaped_weights = weights.reshape(weights.shape[0], -1)  # Flatten each filter

      # Now group weights into sets of 64
      # For example, reshape into groups of (3, 32, 64), each group has 64 weights
      weights = layer.weight.detach().cpu().numpy()  # Shape: (32, 16, 3)

      # Reshape based on dimension of the weights
      reshaped_weights = reshaped_weights.reshape(weights.shape[2],
                                                  weights.shape[2],
                                                  weights.shape[1],
                                                  weights.shape[0])

      # Convert to a list of lists
      reshaped_weights = reshaped_weights.tolist()
      biases = layer.bias.detach().cpu().numpy().tolist()
      return [reshaped_weights, biases]

    # For unsupported layers, return an empty list
    return []


# Function to output shape
def get_outshape(layer, batch_size):
    # LSTM and GRU layer
    if isinstance(layer, (nn.LSTM, nn.GRU)):
        return layer.hidden_size

    # Conv1D layer
    elif isinstance(layer, nn.Conv1d):
        return layer.out_channels

    # Dense layer
    elif isinstance(layer, nn.Linear):
        return layer.out_features

    # PReLU layer (channel size)
    elif isinstance(layer, nn.PReLU):
      return layer.num_parameters

    # BatchNorm1D layer
    elif isinstance(layer, nn.BatchNorm1d):
      # Return num_features for BatchNorm1d
      return layer.num_features

    # BatchNorm2D layer
    elif isinstance(layer, nn.BatchNorm2d):
      # Return (batch_size, num_features) for BatchNorm2d
      return batch_size, layer.num_features

    # Conv2D layer
    elif isinstance(layer, nn.Conv2d):
        return batch_size, layer.out_channels

    # Activation layer
    elif isinstance(layer, (nn.Tanh, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.ELU)):
      # Dependent on previous outshape
      return None


    # Return None or raise an error for unknown layer types
    return None

# Function to create model JSON
def save_model_json(model, input_shape, layers_to_skip=[]):
    # Determine the layer name
    def get_layer_name(i, layer):
      if hasattr(model, 'layer_names'):
        if model.layer_names[i] is not None:
          return model.layer_names[i]
      else:
        return layer_name

    # Determine the layer type
    def get_layer_type(layer):
      if isinstance(layer, nn.LSTM):
            return "lstm"

      if isinstance(layer, nn.GRU):
            return "gru"

      if isinstance(layer, nn.Linear):
          return "dense"

      if isinstance(layer, nn.Conv1d):
            return "conv1d"

      elif isinstance(layer, nn.PReLU):
        return "prelu"

      # BatchNorm layers
      elif isinstance(layer, nn.BatchNorm1d):
        return "batchnorm"
      elif isinstance(layer, nn.BatchNorm2d):
        return "batchnorm2d"

      # Conv2D layer
      elif isinstance(layer, nn.Conv2d):
            return "conv2d"

      # Activation layer
      elif isinstance(layer, (nn.Tanh, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.ELU)):
          return "activation"

      return "unknown"

    def get_layer_activation(layer):
      activation = None
      if isinstance(layer, nn.LSTM):
        if hasattr(model, 'lstm_activation'):
          activation = model.lstm_activation
        else:
          activation = nn.Tanh()
      elif isinstance(layer, nn.GRU):
        if hasattr(model, 'gru_activation'):
          activation = model.gru_activation
        else:
          activation = nn.Tanh()
      elif isinstance(layer, nn.Linear):
        if hasattr(model, 'dense_activation'):
          activation = model.dense_activation
      elif isinstance(layer, nn.Conv1d):
        if hasattr(model, 'conv1d_activation'):
          activation = model.conv1d_activation
      elif isinstance(layer, nn.BatchNorm1d):
        if hasattr(model, 'batch_norm_activation'):
          activation = model.batch_norm_activation
      elif isinstance(layer, nn.BatchNorm2d):
        if hasattr(model, 'batch_norm_2d_activation'):
          activation = model.batch_norm_2d_activation
      elif isinstance(layer, nn.Conv2d):
        if hasattr(model, 'conv2d_activation'):
          activation = model.conv2d_activation
      elif isinstance(layer, (nn.Tanh, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.ELU)):
        if hasattr(model, 'activation'):
          activation = model.activation

      # Recursively handle TimeDistributed-style layers, if they exist in a
      # PyTorch model
      elif isinstance(layer, nn.Sequential):
          for sublayer in layer.children():
              activation = get_layer_activation(sublayer)
              if activation:
                  return activation

      # Check for activation layers
      if isinstance(activation, nn.Tanh):
          return "tanh"
      elif isinstance(activation, nn.ReLU):
          return "relu"
      elif isinstance(activation, nn.Sigmoid):
          return "sigmoid"
      elif isinstance(activation, nn.Softmax):
          return "softmax"
      elif isinstance(activation, nn.ELU):
          return "elu"

      # Return empty if no activation is found or applicable
      return ""

    # Function to save information about each layer
    def save_layer(i, layer):

      # Handle layer-specific attributes
      layer_dict = {
          "name"       : get_layer_name(i, layer),
          "type"       : get_layer_type(layer),
          "activation" : get_layer_activation(layer),
          # Check shape, specifically second component
          "shape"      : [None, input_shape[0], get_outshape(layer, input_shape[0])], # input_shape[0] should actually be the batch size
      }

      # Add Conv1D-specific attributes
      if isinstance(layer, nn.Conv1d):
        layer_dict["kernel_size"] = list(layer.kernel_size)
        layer_dict["dilation"] = list(layer.dilation)
        layer_dict["groups"] = layer.groups

      # Add batch normalization layer attributes
      if isinstance(layer, (nn.BatchNorm1d, nn.BatchNorm2d)):
        layer_dict["epsilon"] = layer.eps
      if isinstance(layer, nn.BatchNorm2d):
        layer_dict["num_filters_in"] = layer.num_features
        layer_dict["num_features_in"] = input_shape[0]

      # Add Conv2D-specific attributes
      if isinstance(layer, nn.Conv2d):
            layer_dict["kernel_size_time"] = layer.kernel_size[0]
            layer_dict["kernel_size_feature"] = layer.kernel_size[1]
            layer_dict["dilation"] = layer.dilation[0] # only time axis
            layer_dict["strides"] = layer.stride[1] # only feature axis
            layer_dict["num_filters_in"] = layer.in_channels
            layer_dict["num_features_in"] = input_shape[0]
            layer_dict["num_filters_out"] = layer.out_channels
            layer_dict["padding"] = str(layer.padding).lower()

      # Flatten the shape if it is a tuple
      shape = layer_dict["shape"]
      layer_dict["shape"] = [item for sublist in shape for item in
       (sublist if isinstance(sublist, tuple) else [sublist])]

      layer_dict["weights"] = get_weights(layer)

      return layer_dict

    model_dict = {}
    # Assuming batch-first input shape
    model_dict["in_shape"] = [None] + list(input_shape)
    layers = []

    # Current shape after each layer to pass to the next one
    current_shape = input_shape

    # Loop over the model's layers and get information (skip the first one)
    for i, (layer_name, layer) in enumerate(model.named_children()):
      # Skip unwanted activation functions
      if layer_name in layers_to_skip or layer_name.endswith('_activation'):
          print(f"Skipping layer: {layer_name}")
          continue

      layer_dict = save_layer(i, layer)
      # Determine output shape and update `current_shape`
      out_shape = get_outshape(layer, batch_size=input_shape[0])
      if get_outshape(layer, input_shape[0]) is not None:
        current_shape = out_shape  # Update shape for the next layer

      # Use previous output shapes for activation layers
      elif isinstance(layer, (nn.Tanh, nn.ReLU, nn.Sigmoid, nn.Softmax, nn.ELU)):
        if isinstance(current_shape, tuple):
          layer_dict["shape"] = [None, input_shape[0]] + list(current_shape)  # Unpack tuple into list
        else:
          layer_dict["shape"] = [None, input_shape[0], current_shape]

      layers.append(layer_dict)

    model_dict["layers"] = layers
    return model_dict

def save_model(model, filename, input_shape, layers_to_skip=[]):
    model_dict = save_model_json(model, input_shape, layers_to_skip)
    with open(filename, "w") as outfile:
        json.dump(model_dict, outfile, indent=4)

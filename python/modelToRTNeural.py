#!/usr/bin/env python3
"""
Converts a recurrent model trained with PyTorch into RTNeural's json format.

The expected input is the json representation of a PyTorch `state_dict` for a
single-input/single-output recurrent network made up of one or more LSTM or GRU
layers followed by a linear layer, as exported by training frameworks such as
Automated-GuitarAmpModelling:

    {
      "model_data": { "model": "SimpleRNN", "input_size": 1, "output_size": 1,
                      "unit_type": "LSTM", "num_layers": 1, "hidden_size": 12,
                      "skip": 1, "bias_fl": true },
      "state_dict": { "rec.weight_ih_l0": [...], "rec.weight_hh_l0": [...],
                      "rec.bias_ih_l0": [...], "rec.bias_hh_l0": [...],
                      "lin.weight": [...], "lin.bias": [...] }
    }

Two conversions are needed to get from PyTorch to RTNeural:

  - RTNeural expects weight matrices in the Keras/TensorFlow layout, which is
    the transpose of the PyTorch layout.
  - PyTorch orders the GRU gates as (reset, update, new), while Keras orders
    them as (update, reset, new), so the gate blocks must be swapped. LSTM gate
    ordering is the same in both frameworks.

PyTorch also splits the recurrent bias into an input and a hidden part. For
LSTM the two can simply be summed; for GRU they must be kept separate, because
the hidden bias of the "new" gate is applied after the reset gate multiplies it.

Usage:
    python3 python/modelToRTNeural.py model.json model_rtneural.json
"""

import argparse
import json

import numpy as np


def convert_numpy_to_list(obj):
    """Recursively convert numpy arrays into plain lists, so they are json serialisable."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    if isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    return obj


def swap_gru_gates(values, hidden_size):
    """Reorder the GRU gate blocks of the last axis from PyTorch's to Keras' order."""
    reset = values[..., 0:hidden_size]
    update = values[..., hidden_size : 2 * hidden_size]
    new = values[..., 2 * hidden_size :]
    return np.concatenate((update, reset, new), axis=-1)


def convert_recurrent_layer(unit_type, hidden_size, w_ih, w_hh, b_ih, b_hh):
    w_ih = np.transpose(w_ih)
    w_hh = np.transpose(w_hh)

    if unit_type == "LSTM":
        return {
            "type": "lstm",
            "activation": "",
            "shape": [None, None, hidden_size],
            "weights": [w_ih, w_hh, b_ih + b_hh],
        }

    if unit_type == "GRU":
        return {
            "type": "gru",
            "activation": "",
            "shape": [None, None, hidden_size],
            "weights": [
                swap_gru_gates(w_ih, hidden_size),
                swap_gru_gates(w_hh, hidden_size),
                np.stack(
                    (
                        swap_gru_gates(b_ih, hidden_size),
                        swap_gru_gates(b_hh, hidden_size),
                    )
                ),
            ],
        }

    raise ValueError("Unsupported unit type: " + str(unit_type))


def convert(model_data, verbose=False):
    try:
        config = model_data["model_data"]
        state_dict = model_data["state_dict"]

        input_size = config["input_size"]
        num_layers = config["num_layers"]
        unit_type = config["unit_type"]
        hidden_size = config["hidden_size"]
        skip = int(config.get("skip", 0))

        lin_weight = np.array(state_dict["lin.weight"])
        lin_bias = np.array(state_dict["lin.bias"])
    except KeyError as error:
        raise ValueError("Missing field in model file: " + str(error))

    if verbose:
        print(
            "Converting {} model: {} layer(s), hidden size {}".format(
                unit_type, num_layers, hidden_size
            )
        )

    rtneural_model = {"in_shape": [None, None, input_size], "layers": []}

    # RTNeural has no notion of an input skip connection, but the information is
    # preserved here so that host applications can re-apply it if they need to.
    if skip > 0:
        rtneural_model["in_skip"] = skip

    for layer in range(num_layers):
        rtneural_model["layers"].append(
            convert_recurrent_layer(
                unit_type,
                hidden_size,
                np.array(state_dict["rec.weight_ih_l%d" % layer]),
                np.array(state_dict["rec.weight_hh_l%d" % layer]),
                np.array(state_dict["rec.bias_ih_l%d" % layer]),
                np.array(state_dict["rec.bias_hh_l%d" % layer]),
            )
        )

    rtneural_model["layers"].append(
        {
            "type": "dense",
            "activation": "",
            "shape": [None, None, lin_weight.shape[0]],
            "weights": [np.transpose(lin_weight), lin_bias],
        }
    )

    return convert_numpy_to_list(rtneural_model)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("model", help="json model file as exported from PyTorch")
    parser.add_argument("output", help="json model file to write, in RTNeural's format")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable verbose output")
    args = parser.parse_args()

    with open(args.model) as json_file:
        model_data = json.load(json_file)

    rtneural_model = convert(model_data, verbose=args.verbose)

    with open(args.output, "w") as json_file:
        json.dump(rtneural_model, json_file, indent=4)

    if args.verbose:
        print("Model saved to " + args.output)


if __name__ == "__main__":
    main()

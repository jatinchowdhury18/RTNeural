#!/usr/bin/env python3
"""Regenerate the reference output for a PyTorch-trained recurrent model.

Reads the json representation of the PyTorch state_dict, rebuilds the network
with torch in double precision, and writes the input/output signals as CSV.

Running in double precision matters: the model is recurrent, so the rounding
error of a single-precision reference accumulates over the test signal and puts
a floor on how closely any other implementation can be shown to match it.
"""

import argparse
import json

import numpy as np
import torch


def build(model_data, dtype=torch.float64):
    config = model_data["model_data"]
    state = model_data["state_dict"]

    unit_type = config["unit_type"]
    rec_cls = {"LSTM": torch.nn.LSTM, "GRU": torch.nn.GRU}[unit_type]

    rec = rec_cls(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dtype=dtype,
    )
    lin = torch.nn.Linear(config["hidden_size"], config["output_size"], dtype=dtype)

    with torch.no_grad():
        for layer in range(config["num_layers"]):
            getattr(rec, "weight_ih_l%d" % layer).copy_(
                torch.tensor(state["rec.weight_ih_l%d" % layer], dtype=dtype)
            )
            getattr(rec, "weight_hh_l%d" % layer).copy_(
                torch.tensor(state["rec.weight_hh_l%d" % layer], dtype=dtype)
            )
            getattr(rec, "bias_ih_l%d" % layer).copy_(
                torch.tensor(state["rec.bias_ih_l%d" % layer], dtype=dtype)
            )
            getattr(rec, "bias_hh_l%d" % layer).copy_(
                torch.tensor(state["rec.bias_hh_l%d" % layer], dtype=dtype)
            )
        lin.weight.copy_(torch.tensor(state["lin.weight"], dtype=dtype))
        lin.bias.copy_(torch.tensor(state["lin.bias"], dtype=dtype))

    return rec, lin, int(config.get("skip", 0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--input", help="existing input csv to reuse", default=None)
    parser.add_argument("--input-raw", help="raw float32 input to use instead", default=None)
    parser.add_argument("--out-x", default="pytorch_x.csv")
    parser.add_argument("--out-y", default="pytorch_y.csv")
    parser.add_argument("--apply-skip", action="store_true", help="add the input skip connection to the output")
    parser.add_argument("--samples", type=int, default=0, help="truncate the input to this many samples")
    args = parser.parse_args()

    with open(args.model) as f:
        model_data = json.load(f)

    rec, lin, skip = build(model_data)
    print("model: {} x{} hidden {}, skip {}".format(
        model_data["model_data"]["unit_type"],
        model_data["model_data"]["num_layers"],
        model_data["model_data"]["hidden_size"],
        skip,
    ))

    if args.input is not None:
        x = np.loadtxt(args.input, delimiter=",")
    else:
        x = np.fromfile(args.input_raw, dtype="<f4").astype(np.float64)

    if args.samples:
        x = x[: args.samples]

    xt = torch.tensor(x, dtype=torch.float64).reshape(-1, 1, 1)
    with torch.no_grad():
        y = lin(rec(xt)[0])
        if args.apply_skip and skip > 0:
            y = y + xt[:, :, :skip]
    y = y.numpy().reshape(-1)

    np.savetxt(args.out_x, x, delimiter=",", fmt="%.18e")
    np.savetxt(args.out_y, y, delimiter=",", fmt="%.18e")
    print("wrote {} samples".format(len(y)))


if __name__ == "__main__":
    main()

"""
Adapted from https://github.com/csteinmetz1/micro-tcn

Licensed under Apache License 2.0
https://github.com/csteinmetz1/micro-tcn/blob/3e1067bcaf07e4ecea88ae16e55437024d1d7eb6/LICENSE

Typically used for real-time modeling of neural audio effects
"""

import json
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn

np.random.seed(1001)
torch.manual_seed(0)

_models = Path("models")
_microtcn = _models / "microtcn"


if not _microtcn.exists():
    _microtcn.mkdir()


class EncodeTensor(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(json.NpEncoder, self).default(obj)


def causal_crop(x, length):
    start = x.shape[-1] - length
    return x[..., start:]


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            dilation=dilation,
            groups=1,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.PReLU(out_ch)

        self.res = nn.Conv1d(in_ch, out_ch, kernel_size=1, groups=in_ch, bias=False)

    def forward(self, x):
        x_in = x

        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)

        x_res = self.res(x_in)

        x = x + causal_crop(x_res, x.shape[-1])

        return x

    def export(self):
        bn_dict = self.bn.state_dict()
        bn_dict["eps"] = self.bn.eps

        layers = [
            ("conv1", self.conv1.state_dict()),
            ("bn", bn_dict),
            ("relu", self.relu.state_dict()),
            ("res", self.res.state_dict()),
        ]

        for layer_name, layer_weight in layers:
            layer_file = _microtcn / Path(layer_name).with_suffix(".json")
            with layer_file.open("w") as json_file:
                json.dump(layer_weight, json_file, cls=EncodeTensor)


if __name__ == "__main__":
    # !!!IMPORTANT!!!
    # Make sure to use `eval` and `no_grad` such that the
    # `running_mean` and `running_var` is not updated after
    # running inference for `y`.
    f = TCNBlock(1, 32, 4, 10).to(torch.float64).eval()
    x = torch.from_numpy(np.random.uniform(-1, 1, 1000)).reshape(1, 1, -1)
    with torch.no_grad():
        y = f(x).detach().numpy()

    np.savetxt("test_data/microtcn_x.csv", x[0].T, delimiter=",")
    np.savetxt("test_data/microtcn_y.csv", y[0], delimiter=",")

    f.export()

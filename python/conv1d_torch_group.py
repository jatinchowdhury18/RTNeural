import json
import numpy as np
import torch
from torch import nn

np.random.seed(1001)
torch.manual_seed(0)

f = nn.Conv1d(in_channels=6, out_channels=3, kernel_size=1, groups=3, dtype=torch.float64)
x = torch.from_numpy(np.random.uniform(-1, 1, 3000)).reshape(1, 6, -1)
y = f(x).detach().numpy()

np.savetxt("test_data/conv1d_torch_group_x_python.csv", x[0].T, delimiter=",")
np.savetxt("test_data/conv1d_torch_group_y_python.csv", y[0], delimiter=",")

class EncodeTensor(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(json.NpEncoder, self).default(obj)

with open("models/conv1d_torch_group.json", "w") as json_file:
    json.dump(f.state_dict(), json_file, cls=EncodeTensor)

import json
import numpy as np
import torch
from torch import nn

np.random.seed(1001)
torch.manual_seed(0)

f = nn.Conv1d(in_channels=6, out_channels=3, kernel_size=3, groups=3, dtype=torch.float64)
g = nn.Conv1d(in_channels=6, out_channels=3, kernel_size=4, dilation=10, groups=3, dtype=torch.float64)
h = nn.Conv1d(in_channels=6, out_channels=6, kernel_size=1, groups=6, dtype=torch.float64)
x = torch.from_numpy(np.random.uniform(-1, 1, 3000)).reshape(1, 6, -1)
y = f(x).detach().numpy()
z = g(x).detach().numpy()
a = h(x).detach().numpy()

np.savetxt("test_data/conv1d_torch_group_x_python.csv", x[0].T, delimiter=",")
np.savetxt("test_data/conv1d_torch_group_y_python_6_3_3_1_3.csv", y[0], delimiter=",")
np.savetxt("test_data/conv1d_torch_group_y_python_6_3_4_10_3.csv", z[0], delimiter=",")
np.savetxt("test_data/conv1d_torch_group_y_python_6_6_1_1_6.csv", a[0], delimiter=",")

class EncodeTensor(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(json.NpEncoder, self).default(obj)

with open("models/conv1d_torch_group_6_3_3_1_3.json", "w") as json_file:
    json.dump(f.state_dict(), json_file, cls=EncodeTensor)

with open("models/conv1d_torch_group_6_3_4_10_3.json", "w") as json_file:
    json.dump(g.state_dict(), json_file, cls=EncodeTensor)

with open("models/conv1d_torch_group_6_6_1_1_6.json", "w") as json_file:
    json.dump(h.state_dict(), json_file, cls=EncodeTensor)

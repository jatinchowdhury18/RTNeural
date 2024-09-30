import torch
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import json
from json import JSONEncoder

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(json.NpEncoder, self).default(obj)

np.random.seed(1001)
torch.manual_seed(0)
out_channels = 12
x = np.random.uniform(-1, 1, 1000)
# Fails if it doesn't have bias
conv = torch.nn.Conv1d(1, out_channels, 5, dilation=1, stride=3, padding='valid', bias=True, dtype=torch.float64)
y = conv(torch.from_numpy(x).reshape(1, 1, -1)).detach().numpy()[0]

print(torch.from_numpy(x).reshape(1, 1, -1).shape)
print(np.shape(y))

plt.plot(x)
plt.plot(y[0, :])
# plt.show()

np.savetxt('test_data/conv1d_torch_x_python_stride_3.csv', x, delimiter=',')
np.savetxt('test_data/conv1d_torch_y_python_stride_3.csv', y, delimiter=',')

with open('models/conv1d_torch_stride_3.json', 'w') as json_file:
    json.dump(conv.state_dict(), json_file,cls=EncodeTensor)

# print(x[:5])
# print(conv.state_dict())
#
# ch_idx = 0
# kernel_test = conv.state_dict()["weight"][ch_idx, 0, :].detach().numpy()
# print(kernel_test)
# y_test = np.correlate(x, kernel_test, mode='full')
# print(y_test[:10])
# print(y[ch_idx,:10])
#
# print(np.sum(kernel_test * x[:5]))

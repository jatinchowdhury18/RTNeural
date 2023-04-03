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

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(1, 8)
        self.dense = torch.nn.Linear(8, 1)

    def forward(self, torch_in):
        x, _ = self.lstm(torch_in)
        return self.dense(x)

x = np.random.uniform(-1, 1, 1000)
torch_in = torch.from_numpy(x.astype(np.float32)).reshape(-1, 1)

model = Model()
y = model.forward(torch_in).detach().numpy()

print(np.shape(y))

plt.plot(x)
plt.plot(y[0, :])
plt.show()

np.savetxt('test_data/lstm_torch_x_python.csv', x, delimiter=',')
np.savetxt('test_data/lstm_torch_y_python.csv', y, delimiter=',')

with open('models/lstm_torch.json', 'w') as json_file:
    json.dump(model.state_dict(), json_file,cls=EncodeTensor)

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

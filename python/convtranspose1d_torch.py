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

in_channels = 4
out_channels = 15
kernel_size = 5
padding = 3
dilation = 1
stride = 3
output_padding = 0
x = torch.tensor(np.random.uniform(-1, 1, [in_channels,100])).unsqueeze(0)

in_length = x.shape[-1]
conv = torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, 
                                dilation=dilation, padding=padding,
                                output_padding=output_padding, stride=stride,
                                bias=True, dtype=torch.float64)

y = conv(x).detach().numpy()[0]

# print('x',x.shape)
# print('y',y.shape)
# print('Before sliding the kernel, data is zero padded:',dilation*(kernel_size -1) - padding,'units in both sides.')
# print('Expected Length:',(in_length-1)*stride-2*padding+dilation*(kernel_size -1)+output_padding+1)
# plt.show()

np.savetxt('test_data/convtranspose1d_torch_x_python.csv', x.squeeze(0), delimiter=',')
np.savetxt('test_data/convtranspose1d_torch_y_python.csv', y, delimiter=',')

with open('models/convtranspose1d_torch.json', 'w') as json_file:
    json.dump(conv.state_dict(), json_file,cls=EncodeTensor)

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt
from model_utils import save_model

# construct TensorFlow model
model = keras.Sequential()
model.add(keras.layers.InputLayer(1))
model.add(keras.layers.Dense(8, kernel_initializer='random_normal', bias_initializer='random_normal'))
model.add(keras.layers.Activation('tanh'))
model.add(keras.layers.Dense(8, kernel_initializer='orthogonal', bias_initializer='random_normal'))
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Dense(8, kernel_initializer='orthogonal', bias_initializer='random_normal'))
model.add(keras.layers.Activation('elu'))
model.add(keras.layers.Dense(8, kernel_initializer='orthogonal', bias_initializer='random_normal'))
model.add(keras.layers.Activation('softmax'))
model.add(keras.layers.Dense(1, kernel_initializer='orthogonal', bias_initializer='random_normal'))

# construct signals
N = 100
x = 500 * np.sin(np.arange(N) * np.pi * 0.1)
y = model(x.reshape(-1, 1))
y = y.numpy()

# plot signals
plt.figure()
plt.plot(x)
plt.plot(y, '--')
plt.ylim(-1.1, 1.1)
plt.savefig('python/dense.png')

# save signals
np.savetxt('test_data/dense_x_python.csv', x, delimiter=',')
np.savetxt('test_data/dense_y_python.csv', y, delimiter=',')

save_model(model, 'models/dense.json')

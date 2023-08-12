import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Conv1D, PReLU, BatchNormalization
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt
from model_utils import save_model

input_shape = (128, 1)

x = Input(shape=input_shape,name = "x")
conv1 = Conv1D(filters=12, kernel_size=65, strides=1, dilation_rate=1, activation=None, padding='valid',name = "conv1")(x)
PRelu1 = PReLU(alpha_initializer='glorot_uniform', shared_axes=[1],name = "PRelu1")(conv1)
bn1 = BatchNormalization(momentum=0.0, epsilon=0.01, beta_initializer='random_normal', gamma_initializer='glorot_uniform', moving_mean_initializer="random_normal", moving_variance_initializer="ones",name = "bn1")(PRelu1)
conv2 = Conv1D(filters=8, kernel_size=33, strides=1, dilation_rate=1, activation=None, padding='valid',name = "conv2")(bn1)
PRelu2 = PReLU(alpha_initializer='glorot_uniform', shared_axes=[1],name = "PRelu2")(conv2)
bn2 = BatchNormalization(momentum=0.0, epsilon=0.01, beta_initializer='random_normal', gamma_initializer='glorot_uniform', moving_mean_initializer="random_normal", moving_variance_initializer="ones",name = "bn2")(PRelu2)
conv3 = Conv1D(filters=4, kernel_size=13, strides=1, dilation_rate=1, activation=None, padding='valid',name = "conv3")(bn2)
PRelu3 = PReLU(alpha_initializer='glorot_uniform', shared_axes=[1],name = "PRelu3")(conv3)
bn3 = BatchNormalization(momentum=0.0, epsilon=0.01, beta_initializer='random_normal', gamma_initializer='glorot_uniform', moving_mean_initializer="random_normal", moving_variance_initializer="ones",name = "bn3")(PRelu3)
conv4 = Conv1D(filters=1, kernel_size=5, strides=1, dilation_rate=1, activation="tanh", padding='valid',name = "conv4")(bn3)

model = keras.Model(inputs=x, outputs=conv4)
model.summary()

# construct signals
x_data = 10 * np.sin(np.arange(input_shape[0]) * np.pi * 0.1)
y = model.predict((x_data.reshape((1, -1, 1))))
print(y.shape)
y = y.flatten()

# save signals
np.savetxt('test_data/conv_stateless_x_python.csv', x_data, delimiter=',')
np.savetxt('test_data/conv_stateless_y_python.csv', y, delimiter=',')

save_model(model, 'models/conv_stateless.json')

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
from tensorflow import keras
tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt
from model_utils import save_model

n_frames = 1
n_features = 10

# construct TensorFlow model
model = keras.Sequential()
model.add(keras.layers.InputLayer(input_shape=(n_frames, n_features, 1)))
# model.add(keras.layers.Conv2D(1, (1, 1), dilation_rate=(1, 1), strides=(1, 1), padding='valid', kernel_initializer='random_normal', bias_initializer='random_normal'))

model.add(keras.layers.Conv2D(1, (1, 3), dilation_rate=(1, 1), strides=(1, 3), padding='same', kernel_initializer='random_normal', bias_initializer='zeros'))
# model.add(keras.layers.BatchNormalization(beta_initializer='random_normal', gamma_initializer='glorot_uniform', moving_mean_initializer="random_normal", moving_variance_initializer="ones"))
# model.add(keras.layers.ReLU())
# model.add(keras.layers.Conv2D(1, (5, 7), dilation_rate=(1, 1), strides=(1, 1), padding='same', kernel_initializer='random_normal', bias_initializer='random_normal'))
# model.add(keras.layers.BatchNormalization(beta_initializer='random_normal', gamma_initializer='glorot_uniform', moving_mean_initializer="random_normal", moving_variance_initializer="ones"))
# model.add(keras.layers.Conv2D(1, (5, 7), dilation_rate=(1, 1), strides=(1, 2), padding='valid', kernel_initializer='random_normal', bias_initializer='random_normal'))
# model.add(keras.layers.BatchNormalization(beta_initializer='random_normal', gamma_initializer='glorot_uniform', moving_mean_initializer="random_normal", moving_variance_initializer="ones"))
# model.add(keras.layers.ReLU())
model.summary()

# construct signals
x = np.sin(np.arange(n_frames * n_features)).reshape((1, n_frames, n_features, 1))
y = model.predict(x)

# save signals
np.savetxt('test_data/conv2d_x_python.csv', x.flatten(), delimiter=',')
np.savetxt('test_data/conv2d_y_python.csv', y.flatten(), delimiter=',')

save_model(model, 'models/conv2d.json')
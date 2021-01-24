# RTNeural

![Tests](https://github.com/jatinchowdhury18/RTNeural/workflows/Tests/badge.svg)

A lightweight neural network inferencing engine written in C++.
This library was designed with the intention of being used in
real-time audio processing, but may be useful for other tasks
as well.

Currently supported layers:
  
  - [x] dense
  - [x] GRU
  - [ ] LSTM
  - [ ] Conv1D

Currently supported activations:
  - [x] tanh
  - [x] ReLU
  - [ ] SoftMax

## How To Use

`RTNeural` is capable of taking a neural network that
has already been trained, loading the weights from that
network, and running inference.

### Exporting weights from a trained network

Neural networks are typically trained using `Python`
libraries including Tensorflow or PyTorch. Once you
have trained a neural network using one of these frameworks,
you must "export" the network weights to a json file,
so that `RTNeural` can read them. An implementation of
the export process for a Tensorflow model is provided in
`python/model_utils.py`, and can be used as follows.

```python
# import dependencies
import tensorflow as tf
from tensorflow import keras
from model_utils import save_model

# create Tensrflow model
model = keras.Sequential()
...

# train model
model.train()

# export model weights
save_model(model, 'model_weights.json')
```

### Creating a model

Next, you can create an inferencing engine in C++ directly
from the exported json file:

```cpp
#include <RTNeural.h>
...
std::ifstream jsonStream("model_weights.json", std::ifstream::binary);
auto model = RTNeural::json_parser::parseJson<double>(jsonStream);
```

### Running inference

Before running inference, it is recommended to "reset" the
state of your model (if the model has state).
```cpp
model->reset();
```

Then, you may run inference as follows:
```cpp
double input[] = { 1.0, 0.5, -0.1 }; // set up input vector
double output = model->forward(input); // compute output
```

## Building with CMake

`RTNeural` is built with CMake, and the easiest way to link
is to use `add_subdirectory(RTNeural)` in your `CMakeLists.txt`.

### Choosing a Backend

`RTNeural` supports three backends,
[`Eigen`](http://eigen.tuxfamily.org/),
[`xsimd`](https://github.com/xtensor-stack/xsimd)
or the C++ STL. You can choose your backend by passing
either `-DRTNEURAL_EIGEN=ON`, `-DRTNEURAL_XSIMD=ON`, or
`-DRTNEURAL_STL=ON` to your CMake configuration. By default,
the `Eigen` backend will be used.

While the `Eigen` backend typically has the best performance,
it is recommended to measure the performance of your network
with all the backends that available on your target platform
to ensure optimal performance. For more information see the
benchmark results (TODO).

### Building the Accuracy Tests

To build the accuracy tests, run
`cmake -Bbuild -DBUILD_TESTS=ON`, followed by
`cmake --build build`. To run the full testing suite,
run `./build/rtneural_tests all`. For more information,
run `./buildrtneural_tests --help`.

### Building the Performance Benchmarks

To build the performance benchmarks, run
`cmake -Bbuild -DBUILD_BENCH=ON`, followed by
`cmake --build build`. To run the layer benchmarks, run
`./build/rtneural_layer_bench <layer> <length> <in_size> <out_size>`.

## License

RTNeural is open source, and is licensed under the
BSD 3-clause license.

Enjoy!

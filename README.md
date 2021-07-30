# RTNeural

[![Tests](https://github.com/jatinchowdhury18/RTNeural/workflows/Tests/badge.svg)](https://github.com/jatinchowdhury18/RTNeural/actions/workflows/tests.yml)
[![Bench](https://github.com/jatinchowdhury18/RTNeural/workflows/Bench/badge.svg)](https://github.com/jatinchowdhury18/RTNeural/actions/workflows/bench.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2106.03037-b31b1b.svg)](https://arxiv.org/abs/2106.03037)
[![License](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A lightweight neural network inferencing engine written in C++.
This library was designed with the intention of being used in
real-time systems, specifically real-time audio processing.

Currently supported layers:
  
  - [x] dense
  - [x] GRU
  - [x] LSTM
  - [x] Conv1D
  - [ ] MaxPooling
  - [ ] BatchNorm

Currently supported activations:
  - [x] tanh
  - [x] ReLU
  - [x] Sigmoid
  - [x] SoftMax

For a complete reference of the available functionality,
see the [API docs](https://ccrma.stanford.edu/~jatin/chowdsp/RTNeural).

## How To Use

`RTNeural` is capable of taking a neural network that
has already been trained, loading the weights from that
network, and running inference. An example of using
RTNeural within an audio plugin can be found on GitHub
[here](https://github.com/jatinchowdhury18/RTNeural-example).

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

### Compile-Time API

The code shown above will create the inferencing engine
dynamically at run-time. If the model architecture is
fixed at compile-time, it may be preferable to use RTNeural's
API for defining an inferencing engine type at compile-time,
which can significantly improve performance.
```cpp
// define model type
RTNeural::ModelT<double, 8, 1
    RTNeural::DenseT<double, 8, 8>,
    RTNeural::TanhActivationT<double, 8>,
    RTNeural::DenseT<double, 8, 1>
> modelT;

// load model weights from json
std::ifstream jsonStream("model_weights.json", std::ifstream::binary);
auto model = RTNeural::json_parser::parseJson<double>(jsonStream);
modelT.parseJson(jsonStream);

modelT.reset(); // reset state

double input[] = { 1.0, 0.5, -0.1 }; // set up input vector
double output = modelT.forward(input); // compute output
```

## Building with CMake

`RTNeural` is built with CMake, and the easiest way to link
is to include `RTNeural` as a submodule:
```cmake
...
add_subdirectory(RTNeural)
include_directories(RTNeural)
...
target_link_libraries(MyCMakeProject LINK_PUBLIC RTNeural)
```

### Choosing a Backend

`RTNeural` supports three backends,
[`Eigen`](http://eigen.tuxfamily.org/),
[`xsimd`](https://github.com/xtensor-stack/xsimd),
or the C++ STL. You can choose your backend by passing
either `-DRTNEURAL_EIGEN=ON`, `-DRTNEURAL_XSIMD=ON`,
or `-DRTNEURAL_STL=ON` to your CMake configuration. By
default, the `Eigen` backend will be used. Alternatively,
you may select your choice of backends in your CMake 
configuration as follows:
```cmake
set(RTNEURAL_XSIMD ON CACHE BOOL "Use RTNeural with this backend" FORCE)
add_subdirectory(modules/RTNeural)
```

In general, the `Eigen` backend typically has the best 
performance for larger networks, while smaller networks
may perform better with XSIMD. However, it is recommended
to measure the performance of your network with all the 
backends that available on your target platform
to ensure optimal performance. For more information see the
[benchmark results](https://github.com/jatinchowdhury18/RTNeural/actions?query=workflow%3ABench).

RTNeural also has experimental support for Apple's
[`Accelerate`](https://developer.apple.com/documentation/accelerate) framework (`-DRTNEURAL_ACCELERATE=ON`).
Please note that the `Accelerate` backend can only be
used when compiling for Apple devices, and does not
currently support defining [compile-time inferencing
engines](#compile-time-api).

### Building the Unit Tests

To build the RTNeural's unit tests, run
`cmake -Bbuild -DBUILD_TESTS=ON`, followed by
`cmake --build build`. To run the full testing suite,
run `./build/rtneural_tests all`. For more information,
run `./build/rtneural_tests --help`.

### Building the Performance Benchmarks

To build the performance benchmarks, run
`cmake -Bbuild -DBUILD_BENCH=ON`, followed by
`cmake --build build --config Release`. To run the layer benchmarks, run
`./build/rtneural_layer_bench <layer> <length> <in_size> <out_size>`. To
run the model benchmark, run `./build/rtneural_model_bench`.

## Contributing

Contributions to this project are most welcome!
Currently, there is considerable need for the
following improvements:
- Better implementation of convolutional layers:
  - Implement more options (grouping, stride, etc...)
  - Implement Conv2D
- Support for exporting/loading PyTorch models
- More robust support for exporting/loading Tensorflow models
- Support for more activation layers
- Better test coverage
- Any changes that improve overall performance

General code maintenance and documentation is always
appreciated as well! Note that if you are implementing
a new layer type, it is not required to provide support
for all the backends, though it is recommended to at
least provide a "fallback" implementation using the STL
backend.

## Powered by RTNeural

RTNeural is currently being used by several audio plugins:

- [Chow Centaur](https://github.com/jatinchowdhury18/KlonCentaur): A guitar pedal emulation plugin, using a real-time recurrent neural network.
- [Chow Tape Model](https://github.com/jatinchowdhury18/AnalogTapeModel): An analog tape emulation, using a real-time dense neural network.
- [GuitarML](https://guitarml.com/): GuitarML plugins use machine learning to model guitar amplifiers and effects.

## Citation

If you are using RTNeural as part of an academic work, please cite the library as follows:
```
@article{chowdhury2021rtneural,
        title={RTNeural: Fast Neural Inferencing for Real-Time Systems}, 
        author={Jatin Chowdhury},
        year={2021},
        journal={arXiv preprint arXiv:2106.03037}
}
```

## License

RTNeural is open source, and is licensed under the
BSD 3-clause license.

Enjoy!

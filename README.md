<p align=center>
  <picture>
    <img src="./doxygen/logo.png" height="200"/>
  </picture>
</p>

# RTNeural

[![Tests](https://github.com/jatinchowdhury18/RTNeural/workflows/Tests/badge.svg)](https://github.com/jatinchowdhury18/RTNeural/actions/workflows/tests.yml)
[![Bench](https://github.com/jatinchowdhury18/RTNeural/workflows/Bench/badge.svg)](https://github.com/jatinchowdhury18/RTNeural/actions/workflows/bench.yml)
[![Examples](https://github.com/jatinchowdhury18/RTNeural/actions/workflows/examples.yml/badge.svg)](https://github.com/jatinchowdhury18/RTNeural/actions/workflows/examples.yml)
[![codecov](https://codecov.io/gh/jatinchowdhury18/RTNeural/branch/main/graph/badge.svg?token=QBEBVSCQTW)](https://codecov.io/gh/jatinchowdhury18/RTNeural)
[![arXiv](https://img.shields.io/badge/arXiv-2106.03037-b31b1b.svg)](https://arxiv.org/abs/2106.03037)
[![License](https://img.shields.io/badge/License-BSD-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A lightweight neural network inferencing engine written in C++.
This library was designed with the intention of being used in
real-time systems, specifically real-time audio processing.

Currently supported layers:
  
  - [x] Dense
  - [x] GRU
  - [x] LSTM
  - [x] Conv1D
  - [x] Conv2D
  - [ ] MaxPooling
  - [x] BatchNorm1D
  - [x] BatchNorm2D

Currently supported activations:
  - [x] tanh
  - [x] ReLU
  - [x] Sigmoid
  - [x] SoftMax
  - [x] ELu
  - [x] PReLU

For a complete reference of the available functionality,
see the [API docs](https://ccrma.stanford.edu/~jatin/chowdsp/RTNeural).
For more information on the design and purpose of the library,
see the [reference paper](https://arxiv.org/abs/2106.03037).

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

## How To Use

`RTNeural` is capable of taking a neural network that
has already been trained, loading the weights from that
network, and running inference. Some simple examples
are available in the [`examples/`](./examples) directory.

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

For an example of exporting a model from PyTorch,
see [this example script](./python/gru_torch.py).

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
modelT.parseJson(jsonStream);

modelT.reset(); // reset state

double input[] = { 1.0, 0.5, -0.1 }; // set up input vector
double output = modelT.forward(input); // compute output
```

### Loading Layers from PyTorch

The above example code assumes that the trained model has
been exported from TensorFlow. For loading PyTorch models,
the RTNeural namespace `RTNeural::torch_helpers`, provides
helper functions for loading layers exported from PyTorch.

```cpp
// load model weights from json
std::ifstream jsonStream("model_weights.json", std::ifstream::binary);
nlohmann::json modelJson;
jsonStream >> modelJson;

// load a layer from a static model
RTNeural::ModelT<float, 1, 1, RTNeural::DenseT<float, 1, 1>> model;
RTNeural::torch_helpers::loadDense(modelJson, "name_of_layer.", model.get<0>());
```

For more examples, see the
[`examples/torch`](./examples/torch) directory.

## Building with CMake

`RTNeural` is built with CMake, and the easiest way to link
is to include `RTNeural` as a submodule:
```cmake
...
add_subdirectory(RTNeural)
target_link_libraries(MyCMakeProject LINK_PUBLIC RTNeural)
```

If you are trying to use RTNeural in a project that does not use
CMake, please see the [instructions below](#building-without-cmake).

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
backends that are available on your target platform
to ensure optimal performance. For more information see the
[benchmark results](https://github.com/jatinchowdhury18/RTNeural/actions?query=workflow%3ABench).

RTNeural also has experimental support for Apple's
[`Accelerate`](https://developer.apple.com/documentation/accelerate) framework (`-DRTNEURAL_ACCELERATE=ON`).
Please note that the `Accelerate` backend can only be
used when compiling for Apple devices, and does not
currently support defining [compile-time inferencing
engines](#compile-time-api).

Note that you must abide by the licensing rules of whichever backend library you choose.

### Other configuration flags

If you would like to build RTNeural with the AVX SIMD extensions,
you may run CMake with the `-DRTNEURAL_USE_AVX=ON`. Note that
this flag will have no effect when compiling for platforms that
do not support AVX instructions.

### Building the Unit Tests

To build RTNeural's unit tests, run
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

### Building the Examples

To build the RTNeural examples run:
```bash
cmake -Bbuild -DBUILD_EXAMPLES=ON
cmake --build build --config Release
```
The example programs will then be located in
`build/examples_out/`, and may be run from there.

An example of using RTNeural within a real-time
audio plugin can be found on GitHub
[here](https://github.com/jatinchowdhury18/RTNeural-example).

## Building without CMake

If you wish to use RTNeural in a project that doesn't use CMake,
RTNeural can be included as a header-only library, along with a few
extra steps.

1. Add a compile-time definition to define a default byte alignment for RTNeural.
   For most cases this definition will be one of either:
   - `RTNEURAL_DEFAULT_ALIGNMENT=16`
   - `RTNEURAL_DEFAULT_ALIGNMENT=32`

2. Add a compile-time definition to [select a backend](#choosing-a-backend).
   If you wish to use the STL backend, then no definition is required.
   This definition should be one of the following:
   - `RTNEURAL_USE_EIGEN=1`
   - `RTNEURAL_USE_XSIMD=1`

4. Add the necessary include paths for your chosen backend. This path will be
   one of either:
   - `<repo>/modules/Eigen`
   - `<repo>/modules/xsimd/include/xsimd`

It may also be worth checking out the
[example Makefile](./examples/hello_rtneural/Makefile).

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

## Contributors

Please thank the following individuals for their important contributions:

- [wayne-chen](https://github.com/wayne-chen): Softmax activation layer and general API improvements
- [hollance](https://github.com/hollance): RTNeural logo
- [stepanmk](https://github.com/stepanmk): Eigen Conv1D layer optimization
- [DamRsn](https://github.com/DamRsn): Eigen implementations for Conv2D and BatchNorm2D layers

## Powered by RTNeural

RTNeural is currently being used by several audio plugins and other projects:

- [4000DB-NeuralAmp](https://github.com/EnrcDamn/4000DB-NeuralAmp): Neural emulation of the pre-amp section from the Akai 4000DB tape machine.
- [AIDA-X](https://github.com/AidaDSP/AIDA-X): An AU/CLAP/LV2/VST2/VST3 audio plugin that loads RTNeural models and cabinet IRs.
- [BYOD](https://github.com/Chowdhury-DSP/BYOD): A guitar distortion plugin containing several machine learning-based effects.
- [Chow Centaur](https://github.com/jatinchowdhury18/KlonCentaur): A guitar pedal emulation plugin, using a real-time recurrent neural network.
- [Chow Tape Model](https://github.com/jatinchowdhury18/AnalogTapeModel): An analog tape emulation, using a real-time dense neural network.
- [cppTimbreID](https://github.com/domenicostefani/cpp-timbreID): An audio feature extraction library.
- [GuitarML](https://guitarml.com/): GuitarML plugins use machine learning to model guitar amplifiers and effects.
- [MLTerror15](https://github.com/IHorvalds/MLTerror15): Deeply learned simulator for the Orange Tiny Terror with Recurrent Neural Networks.
- [NeuralNote](https://github.com/DamRsn/NeuralNote): An audio-to-MIDI transcription plugin using Spotify's [basic-pitch](https://github.com/spotify/basic-pitch) model.
- [rt-neural-lv2](https://github.com/AidaDSP/aidadsp-lv2): A headless lv2 plugin using RTNeural to model guitar pedals and amplifiers.
- Tone Empire plugins:
  - [LVL - 01](https://tone-empire.com/shop/lvl-01/): An A.I./M.L.-based compressor effect.
  - [TM700](https://tone-empire.com/shop/tm700/): A machine learning tape emulation effect.
  - [Neural Q](https://tone-empire.com/shop/neuralq-v2/): An analog emulation 2-band EQ, using recurrent neural networks.
- [ToobAmp](https://github.com/rerdavies/ToobAmp): Guitar effect plugins for the Raspberry Pi.


If you are using RTNeural in one of your projects, let us know and we will add it to this list!

## License

RTNeural is open source, and is licensed under the
BSD 3-clause license.

Enjoy!

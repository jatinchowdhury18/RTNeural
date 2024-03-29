# RTNeural Examples

This directory contains example programs to
demonstrate usage of the RTNeural library:

- `hello_rtneural`: Loads a model from a file and runs inference.
- `rtneural_static_model`: Demonstrates how to use the RTNeural compile-time API to load and run a static model.
- `rtneural_dynamic_model`: Demonstrates how to use the RTNeural run-time API to load and run a model from a file.
- `custom_layer_model`: Demonstrates how to extend RTNeural's compile-time API with custom layers.
- `torch`: Demonstrates how to use the RTNeural compile-time API to import pytorch models. The exporting from python pytorch of the models used in those examples can be found in [RTNeural/python](../python/). They have a `_torch` postfix.
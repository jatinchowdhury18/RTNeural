algorithms.hpp used to exist, and some functions are used some functions
in RTNeural. Later on, xsimd removed this file from the public API as
they were alternative implementation for C++17/20 standard APIs.
We are not replacing those functions to not forcing newer C++ versions.
Hence importing this file here.
xsimd_api.hpp is the only subsequent dependency.

For more details, see: https://github.com/jatinchowdhury18/RTNeural/pull/81

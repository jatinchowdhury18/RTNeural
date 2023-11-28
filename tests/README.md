# RTNeural Testing

RTNeural tests are configured with CMake's [CTest](https://cmake.org/cmake/help/book/mastering-cmake/chapter/Testing%20With%20CMake%20and%20CTest.html) and developed with the [GoogleTest framework](https://github.com/google/googletest). Tests are split into two categories:

1. unit tests, and
2. functional tests.

A unit test should (ideally) be small, and will usually verify that one piece of
logic in an individual component is working correctly. Functional tests operate
at a higher level, and should check that the combination of multiple components
(or multiple pieces of logic) are operating together as expected.

GoogleTest has excellent [documentation](https://google.github.io/googletest/),
which is highly recommended reading if you're new to testing or new to the
framework.

Happy testing!

## Running tests

### Full test suite

To build and run all tests, make sure that you have passed `-DBUILD_TESTS=ON`
to your `cmake` command. You then have two options:

1. build all targets as normal, then run the test suite by invoking `ctest` in your build folder, or
2. build the `rtneural_test` target, which should also run the test suite.

### All unit tests

To build and run only the unit tests,

1. build the `rtneural_test_unit` target, and
2. run the executable in your build folder at `./tests/unit/rtneural_test_unit`

### All functional tests

To build and run only the functional tests,

1. build the `rtneural_test_functional` target, and
2. run the executable in your build folder at `./tests/functional/rtneural_test_functional`

### A subset of the tests

Running a sub-set of the tests within a test target like `rtneural_test_unit` can be done by passing
a filter string to the `--gtest_filter` command-line argument, e.g.

```sh
./tests/unit/rtneural_test_unit --gtest_filter="*Activation*"
```

There are a wealth of other useful command-line options that can be found by passing the `--help`
argument to the test executable.



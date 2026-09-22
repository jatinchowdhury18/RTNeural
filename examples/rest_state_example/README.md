# Rest state example

A trained network usually does not output zero when it processes silence: it
settles at a non-zero **rest state**. After `model.reset()` every layer starts
from zero instead, so the output has to travel from zero to that rest state,
which is heard as a click or a short swell at the start of playback.

This example captures the rest state once, at build time, and restores it with
`model.reset(state)`, so the output is already settled on the first sample.

## How it works

### At build time: `generate_rest_state`

CMake runs this program automatically before compiling the example, and runs it
again whenever the generator or the model file changes. It:

1. resets the model to zero with `model.reset()`,
2. feeds it 2 s of silence, so it settles,
3. captures the settled state with `model.getState(restState)`,
4. writes that state, plus the DC blocker's rest state (see below), to
   `rest_state.h` as constants:
   ```cpp
   constexpr std::array restState { ... };
   constexpr float dcBlockerRestState = ...;
   ```

### At run time: `rest_state_example`

The example includes `rest_state.h`, so the rest state is already there when it
starts. It then:

1. resets the model with `model.reset(restState.data())`,
2. resets the DC blocker with `dcBlocker.reset(dcBlockerRestState)`,
3. processes audio, with no transient.

To show the difference, it prints the first samples of silence twice, once after
a zero reset and once after the rest-state reset:

```
Zero-state model and DC blocker, feeding silence:
  sample 0: -0.0108399
  sample 1: -1.84899e-05
  sample 2: 0.00903408
  ...
Rest-state model and DC blocker, feeding silence:
  sample 0: 0
  sample 1: 0
  ...
```

## The DC blocker

The network output at rest is a small constant `c` (a DC offset), so the example
also runs a one-pole DC blocker after the model:

```
y[n] = x[n] + s[n-1]
s[n] = R * y[n] - x[n]
```

A filter has its own state, and it needs its own rest state too. With a constant
input `c`, the output settles at `y = 0`, so its single state value settles at
`s = R * 0 - c = -c`, and the generator writes `dcBlockerRestState = -c`.
Starting the filter from zero instead would produce a new transient: `c`,
decaying as `c * R^n`.

## Files

| File | Purpose |
|---|---|
| `rest_state_model.h` | Model type, model loader, sample rate and DC blocker cutoff, shared by both programs |
| `generate_rest_state.cpp` | Build-time generator that writes `rest_state.h` |
| `rest_state_example.cpp` | Uses `rest_state.h` and compares a zero reset with a rest-state reset |
| `dc_blocker.h` | One-pole DC blocker with a single state value |
| `CMakeLists.txt` | Runs the generator and makes the example depend on its output |

The generated header ends up in the build folder, at
`build/examples/rest_state_example/rest_state.h`.

## Building and running

From the repository root:

```sh
cmake -Bbuild -DBUILD_EXAMPLES=ON
cmake --build build --target rest_state_example
./build/examples_out/rest_state_example
```

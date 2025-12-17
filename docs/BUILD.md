# Building Orchard

## Prerequisites

*   macOS with Apple Silicon (M1/M2/M3)
*   Xcode Command Line Tools (or full Xcode) installed
*   CMake (optional, for CMake build)

## Building with Make

1.  Open a terminal in the project root.
2.  Run `make`.
3.  Run `./orchard_test`.

## Building with CMake

1.  Create a build directory: `mkdir build && cd build`
2.  Configure: `cmake ..`
3.  Build: `make`
4.  Run: `./bin/orchard_test` (Note: You might need to copy `matmul.metallib` to the bin directory or run from the root depending on how the path is resolved).

## Troubleshooting

If you see `xcrun: error: unable to find utility "metal"`, ensure that Xcode Command Line Tools are properly installed and selected:

```bash
xcode-select --install
sudo xcode-select --reset
```

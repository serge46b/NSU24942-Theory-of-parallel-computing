# Build instructions

Build with CMake (out-of-source):

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

Run the executable: `./main` (or `main.exe` on Windows).

## USE_DOUBLE

To compile with double-precision output instead of float, enable the option when configuring:

```bash
cmake -DUSE_DOUBLE=ON ..
```

Then build as usual. Omit the flag or use `-DUSE_DOUBLE=OFF` for the default (float).

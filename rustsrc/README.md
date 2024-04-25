# Rust functions for CS336

## Installation

Added some commands below for installation of rust on macos.

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
clang --version # usually has on macOS
brew install oniguruma # for macOS
pip install maturin
cd rustsrc
maturin develop --release
python core/tokenizer.py
```

1. Ensure you have Rust on your machine. You can get it with rustup, using the system package manager, or even with conda.
   `rustc` and `cargo` should be available on your machine.
   Additionally, Oniguruma (the library used for complex regex parsing) requires `clangdev`, which can be gotten by just installing clang on your machine.
   If this is missing, the compiler will fail referencing "ONIG".

2. `maturin` is used to build PyO3 code, so install that using `pip`:

```sh
pip install maturin
```

3. Build and install the Rust code:

```sh
maturin develop --release
```

Builds have been tested extensively on Linux and MacOS.

I'm responsive (on Slack or otherwise), so reach out if there are any questions.

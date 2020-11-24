# OpenCL Demo written in Rusty Mc. Rust

This repository contains demo calculations using opencl with rust.
The goal is to compare the calculation speed to similar applications written with CUDA or running
directly on the CPU.

## Usage

You need a rust toolchain installation and the OpenCL headers.

```sh
USAGE:
    rust-opencl-demo <SUBCOMMAND>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

SUBCOMMANDS:
    calculate-primes    
    help                Prints this message or the help of the given subcommand(s)
```


## License

This project is licensed under Apache 2.0.
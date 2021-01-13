# OpenCL Demo written in Rusty Mc. Rust

This repository contains demo calculations using opencl with rust.
The goal is to compare the calculation speed to similar applications written with CUDA or running
directly on the CPU.

## Usage

You need a rust toolchain installation and the OpenCL headers.

```
USAGE:
    rust-opencl-demo <SUBCOMMAND>

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

SUBCOMMANDS:
    bench-global-size    Benchmarks the global size (number of tasks) value
    bench-local-size     Benchmarks the local size value
    calculate-primes     Calculates primes on the GPU
    help                 Prints this message or the help of the given subcommand(s)
    info                 Prints GPU information
```

### Bench Global Size

```
Benchmarks the global size (number of tasks) value

USAGE:
    rust-opencl-demo bench-global-size [OPTIONS]

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -o, --bench-output <benchmark-file>            The output file for timings [default: bench.csv]
    -n, --calculation-steps <calculation-steps>
            How many calculations steps should be done per GPU thread [default: 1000000]

        --global-size-start <global-size-start>    The start value for the used global size [default: 1024]
        --global-size-step <global-size-step>      The step value for the used global size [default: 128]
        --global-size-stop <global-size-stop>      The stop value for the used global size [default: 1048576]
        --local-size <local-size>                  The maximum number of tasks for the benchmark [default: 128]
    -r, --repetitions <repetitions>
            The average of n runs that is used instead of using one value only. By default the benchmark for each step
            is only run once [default: 1]
```

### Bench Local Size

```
Benchmarks the local size value

USAGE:
    rust-opencl-demo bench-local-size [OPTIONS]

FLAGS:
    -h, --help       Prints help information
    -V, --version    Prints version information

OPTIONS:
    -o, --bench-output <benchmark-file>            The output file for timings [default: bench.csv]
    -n, --calculation-steps <calculation-steps>
            How many calculations steps should be done per GPU thread [default: 1000000]

        --global-size <global-size>                The maximum number of tasks for the benchmark [default: 6144]
        --local-size-start <local-size-start>      The initial number for the local size [default: 4]
        --local-size-step <local-size-step>        The amount the local size increases by every step [default: 4]
        --local-size-stop <local-size-stop>
            The maximum amount of the local size Can't be greater than the maximum local size of the gpu that can be
            retrieved with the info command [default: 1024]
    -r, --repetitions <repetitions>
            The average of n runs that is used instead of using one value only. By default the benchmark for each step
            is only run once [default: 1]
```

### Calculate Primes

```
Calculates primes on the GPU

USAGE:
    rust-opencl-demo calculate-primes [FLAGS] [OPTIONS]

FLAGS:
        --cpu-validate    If the calculated prime numbers should be validated on the cpu by a simple prime algorithm
    -h, --help            Prints help information
        --no-cache        If the prime numbers should be used for the divisibility check instead of using an optimized
                          auto-increment loop
    -V, --version         Prints version information

OPTIONS:
        --local-size <local-size>
            The local size for the tasks. The value for numbers_per_step needs to be divisible by this number. The
            maximum local size depends on the gpu capabilities. If no value is provided, OpenCL chooses it automatically
        --end <max-number>                       The maximum number to calculate to [default: 9223372036854775807]
    -p, --parallel <num-threads>                 number of used threads [default: 2]
        --numbers-per-step <numbers-per-step>
            The amount of numbers that are checked per step. Even numbers are ignored so the Range actually goes to
            numbers_per_step * 2 [default: 33554432]
    -o, --output <output-file>                   The output file for the calculated prime numbers [default: primes.txt]
        --start <start-offset>                   The number to start with [default: 0]
        --timings-output <timings-file>          The output file for timings [default: timings.csv]
```


## License

This project is licensed under Apache 2.0.
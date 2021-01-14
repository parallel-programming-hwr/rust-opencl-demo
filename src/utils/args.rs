/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use std::path::PathBuf;
use structopt::StructOpt;

#[derive(StructOpt, Clone, Debug)]
#[structopt()]
pub enum Opts {
    /// Calculates primes on the GPU
    #[structopt(name = "calculate-primes")]
    CalculatePrimes(CalculatePrimes),

    /// Benchmarks the local size value
    #[structopt(name = "bench-local-size")]
    BenchLocalSize(BenchLocalSize),

    /// Benchmarks the global size (number of tasks) value
    #[structopt(name = "bench-global-size")]
    BenchGlobalSize(BenchGlobalSize),

    /// Prints GPU information
    Info,
}

#[derive(StructOpt, Clone, Debug)]
pub struct CalculatePrimes {
    #[structopt(flatten)]
    pub general_options: GeneralOptions,

    /// The number to start with
    #[structopt(long = "start", default_value = "0")]
    pub start_offset: u64,

    /// The maximum number to calculate to
    #[structopt(long = "end", default_value = "9223372036854775807")]
    pub max_number: u64,

    /// The output file for the calculated prime numbers
    #[structopt(short = "o", long = "output", default_value = "primes.txt")]
    pub output_file: PathBuf,

    /// The output file for timings
    #[structopt(long = "timings-output", default_value = "timings.csv")]
    pub timings_file: PathBuf,

    /// The local size for the tasks.
    /// The value for numbers_per_step needs to be divisible by this number.
    /// The maximum local size depends on the gpu capabilities.
    /// If no value is provided, OpenCL chooses it automatically.
    #[structopt(long = "local-size")]
    pub local_size: Option<usize>,

    /// The amount of numbers that are checked per step. Even numbers are ignored so the
    /// Range actually goes to numbers_per_step * 2.
    #[structopt(long = "numbers-per-step", default_value = "33554432")]
    pub numbers_per_step: usize,

    /// If the prime numbers should be used for the divisibility check instead of using
    /// an optimized auto-increment loop.
    #[structopt(long = "no-cache")]
    pub no_cache: bool,

    /// If the calculated prime numbers should be validated on the cpu by a simple prime algorithm
    #[structopt(long = "cpu-validate")]
    pub cpu_validate: bool,
}

#[derive(StructOpt, Clone, Debug)]
pub struct BenchLocalSize {
    #[structopt(flatten)]
    pub bench_options: BenchOptions,

    /// The initial number for the local size
    #[structopt(long = "local-size-start", default_value = "4")]
    pub local_size_start: usize,

    /// The amount the local size increases by every step
    #[structopt(long = "local-size-step", default_value = "4")]
    pub local_size_step: usize,

    /// The maximum amount of the local size
    /// Can't be greater than the maximum local size of the gpu
    /// that can be retrieved with the info command
    #[structopt(long = "local-size-stop", default_value = "1024")]
    pub local_size_stop: usize,

    /// The maximum number of tasks for the benchmark
    #[structopt(long = "global-size", default_value = "6144")]
    pub global_size: usize,
}

#[derive(StructOpt, Clone, Debug)]
pub struct BenchGlobalSize {
    #[structopt(flatten)]
    pub bench_options: BenchOptions,

    /// The start value for the used global size
    #[structopt(long = "global-size-start", default_value = "1024")]
    pub global_size_start: usize,

    /// The step value for the used global size
    #[structopt(long = "global-size-step", default_value = "128")]
    pub global_size_step: usize,

    /// The stop value for the used global size
    #[structopt(long = "global-size-stop", default_value = "1048576")]
    pub global_size_stop: usize,

    /// The maximum number of tasks for the benchmark
    #[structopt(long = "local-size", default_value = "128")]
    pub local_size: usize,
}

#[derive(StructOpt, Clone, Debug)]
pub struct BenchOptions {
    #[structopt(flatten)]
    pub general_options: GeneralOptions,

    /// How many calculations steps should be done per GPU thread
    #[structopt(short = "n", long = "calculation-steps", default_value = "1000000")]
    pub calculation_steps: u32,

    /// The output file for timings
    #[structopt(short = "o", long = "bench-output", default_value = "bench.csv")]
    pub benchmark_file: PathBuf,

    /// The average of n runs that is used instead of using one value only.
    /// By default the benchmark for each step is only run once
    #[structopt(short = "r", long = "repetitions", default_value = "1")]
    pub repetitions: usize,
}

#[derive(StructOpt, Clone, Debug)]
pub struct GeneralOptions {
    /// If the output should be colored
    #[structopt(long = "color", possible_values = &UseColors::variants(), case_insensitive = true, default_value = "auto")]
    pub color: UseColors,

    /// number of used threads
    #[structopt(short = "p", long = "threads", default_value = "2")]
    pub threads: usize,
}

arg_enum! {
    #[derive(Clone, Debug)]
    pub enum UseColors {
        Off,
        On,
        Auto,
    }
}

/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */

use crate::kernel_controller::primes::is_prime;
use crate::kernel_controller::KernelController;
use crate::output::csv::CSVWriter;
use crate::output::{create_csv_write_thread, create_prime_write_thread};
use rayon::prelude::*;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::mem;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use structopt::StructOpt;

mod kernel_controller;
mod output;

#[derive(StructOpt, Clone, Debug)]
#[structopt()]
enum Opts {
    /// Calculates primes on the GPU
    #[structopt(name = "calculate-primes")]
    CalculatePrimes(CalculatePrimes),

    /// Benchmarks the number of tasks used for the calculations
    #[structopt(name = "bench-task-count")]
    BenchmarkTaskCount(BenchmarkTaskCount),
}

#[derive(StructOpt, Clone, Debug)]
struct CalculatePrimes {
    /// The number to start with
    #[structopt(long = "start", default_value = "0")]
    start_offset: u64,

    /// The maximum number to calculate to
    #[structopt(long = "end", default_value = "9223372036854775807")]
    max_number: u64,

    /// The output file for the calculated prime numbers
    #[structopt(short = "o", long = "output", default_value = "primes.txt")]
    output_file: PathBuf,

    /// The output file for timings
    #[structopt(long = "timings-output", default_value = "timings.csv")]
    timings_file: PathBuf,

    /// The amount of numbers that are checked per step. Even numbers are ignored so the
    /// Range actually goes to numbers_per_step * 2.
    #[structopt(long = "numbers-per-step", default_value = "33554432")]
    numbers_per_step: usize,

    /// If the prime numbers should be used for the divisibility check instead of using
    /// an optimized auto-increment loop.
    #[structopt(long = "no-cache")]
    no_cache: bool,

    /// If the calculated prime numbers should be validated on the cpu by a simple prime algorithm
    #[structopt(long = "cpu-validate")]
    cpu_validate: bool,
}

#[derive(StructOpt, Clone, Debug)]
struct BenchmarkTaskCount {
    /// How many calculations steps should be done per GPU thread
    #[structopt(long = "calculation-steps", default_value = "1000000")]
    calculation_steps: u32,

    /// The initial number of tasks for the benchmark
    #[structopt(long = "num-tasks-start", default_value = "1")]
    num_tasks_start: usize,

    /// The maximum number of tasks for the benchmark
    #[structopt(long = "num-tasks-stop", default_value = "10000000")]
    num_tasks_stop: usize,

    /// The amount the task number increases per step
    #[structopt(long = "num-tasks-step", default_value = "10")]
    num_tasks_step: usize,

    /// The average of n runs that is used instead of using one value only.
    /// By default the benchmark for each step is only run once
    #[structopt(long = "average-of", default_value = "1")]
    average_of: usize,

    /// The output file for timings
    #[structopt(long = "bench-output", default_value = "bench.csv")]
    benchmark_file: PathBuf,
}

fn main() -> ocl::Result<()> {
    let opts: Opts = Opts::from_args();
    let controller = KernelController::new()?;

    match opts {
        Opts::CalculatePrimes(prime_opts) => calculate_primes(prime_opts, controller),
        Opts::BenchmarkTaskCount(bench_opts) => bench_task_count(bench_opts, controller),
    }
}

/// Calculates Prime numbers with GPU acceleration
fn calculate_primes(prime_opts: CalculatePrimes, controller: KernelController) -> ocl::Result<()> {
    let output = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .append(true)
            .open(prime_opts.output_file)
            .unwrap(),
    );
    let timings = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(prime_opts.timings_file)
            .unwrap(),
    );
    let timings = CSVWriter::new(
        timings,
        &[
            "offset",
            "count",
            "gpu_duration",
            "filter_duration",
            "total_duration",
        ],
    )
    .unwrap();

    let (prime_sender, prime_handle) = create_prime_write_thread(output);
    let (csv_sender, csv_handle) = create_csv_write_thread(timings);

    let mut offset = prime_opts.start_offset;
    if offset % 2 == 0 {
        offset += 1;
    }
    if offset < 2 {
        prime_sender.send(vec![2]).unwrap();
    }
    loop {
        let start = Instant::now();
        let numbers = (offset..(prime_opts.numbers_per_step as u64 * 2 + offset))
            .step_by(2)
            .collect::<Vec<u64>>();
        println!(
            "Filtering primes from {} numbers, offset: {}",
            numbers.len(),
            offset
        );
        let prime_result = if prime_opts.no_cache {
            controller.filter_primes_simple(numbers)?
        } else {
            controller.filter_primes(numbers)?
        };
        let primes = prime_result.primes;
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000f64;

        println!(
            "Calculated {} primes in {:.4} ms: {:.4} checks/s",
            primes.len(),
            elapsed_ms,
            prime_opts.numbers_per_step as f64 / start.elapsed().as_secs_f64()
        );
        csv_sender
            .send(vec![
                offset.to_string(),
                primes.len().to_string(),
                duration_to_ms_string(&prime_result.gpu_duration),
                duration_to_ms_string(&prime_result.filter_duration),
                elapsed_ms.to_string(),
            ])
            .unwrap();

        if prime_opts.cpu_validate {
            validate_primes_on_cpu(&primes)
        }
        println!();
        prime_sender.send(primes).unwrap();

        if (prime_opts.numbers_per_step as u128 * 2 + offset as u128)
            > prime_opts.max_number as u128
        {
            break;
        }
        offset += prime_opts.numbers_per_step as u64 * 2;
    }

    mem::drop(prime_sender);
    mem::drop(csv_sender);
    prime_handle.join().unwrap();
    csv_handle.join().unwrap();

    Ok(())
}

fn bench_task_count(opts: BenchmarkTaskCount, controller: KernelController) -> ocl::Result<()> {
    let bench_writer = BufWriter::new(
        OpenOptions::new()
            .truncate(true)
            .write(true)
            .create(true)
            .open(opts.benchmark_file)
            .unwrap(),
    );
    let csv_writer = CSVWriter::new(
        bench_writer,
        &[
            "num_tasks",
            "calc_count",
            "write_duration",
            "gpu_duration",
            "read_duration",
        ],
    )
    .unwrap();
    let (bench_sender, bench_handle) = create_csv_write_thread(csv_writer);
    for n in (opts.num_tasks_start..opts.num_tasks_stop).step_by(opts.num_tasks_step) {
        let mut stats = controller.bench_int(opts.calculation_steps, n)?;
        for _ in 1..opts.average_of {
            stats.avg(controller.bench_int(opts.calculation_steps, n)?)
        }

        println!("{}\n", stats);
        bench_sender
            .send(vec![
                n.to_string(),
                opts.calculation_steps.to_string(),
                duration_to_ms_string(&stats.write_duration),
                duration_to_ms_string(&stats.calc_duration),
                duration_to_ms_string(&stats.read_duration),
            ])
            .unwrap();
    }

    mem::drop(bench_sender);
    bench_handle.join().unwrap();

    Ok(())
}

fn validate_primes_on_cpu(primes: &Vec<u64>) {
    println!("Validating...");
    let failures = primes
        .par_iter()
        .filter(|n| !is_prime(**n))
        .collect::<Vec<&u64>>();
    if failures.len() > 0 {
        println!(
            "{} failures in prime calculation: {:?}",
            failures.len(),
            failures
        );
    } else {
        println!("No failures found.");
    }
}

fn duration_to_ms_string(duration: &Duration) -> String {
    format!("{}", duration.as_secs_f64() * 1000f64)
}

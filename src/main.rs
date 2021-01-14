/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

#[macro_use]
extern crate clap;

mod benching;
mod kernel_controller;
mod output;
mod utils;

use std::fs::{File, OpenOptions};
use std::io::BufWriter;
use std::path::PathBuf;
use std::time::Duration;

use ocl_stream::stream::OCLStream;
use ocl_stream::utils::result::{OCLStreamError, OCLStreamResult};
use rayon::prelude::*;

use structopt::StructOpt;
use utils::args::{BenchGlobalSize, BenchLocalSize, CalculatePrimes, Opts};

use crate::kernel_controller::bench::BenchStatistics;
use crate::kernel_controller::primes::is_prime;
use crate::kernel_controller::KernelController;
use crate::output::csv::ThreadedCSVWriter;
use crate::output::threaded::ThreadedWriter;
use crate::utils::args::UseColors;
use crate::utils::logging::init_logger;
use chrono::Local;

fn main() -> OCLStreamResult<()> {
    let opts: Opts = Opts::from_args();
    let controller = KernelController::new()?;
    init_logger();

    match opts {
        Opts::Info => controller.print_info().map_err(OCLStreamError::from),
        Opts::CalculatePrimes(prime_opts) => calculate_primes(prime_opts, controller),
        Opts::BenchGlobalSize(bench_opts) => bench_global_size(bench_opts, controller),
        Opts::BenchLocalSize(bench_opts) => bench_local_size(bench_opts, controller),
    }
}

/// Calculates primes on the GPU
fn calculate_primes(
    prime_opts: CalculatePrimes,
    mut controller: KernelController,
) -> OCLStreamResult<()> {
    set_output_colored(prime_opts.general_options.color);
    controller.set_concurrency(prime_opts.general_options.threads);

    let csv_file = open_write_buffered(&prime_opts.timings_file);
    let mut csv_writer =
        ThreadedCSVWriter::new(csv_file, &["timestamp", "first", "count", "gpu_duration"]);
    let output_file = open_write_buffered(&prime_opts.output_file);

    let output_writer = ThreadedWriter::new(output_file, |v: Vec<u64>| {
        v.iter()
            .map(|v| v.to_string())
            .fold("".to_string(), |a, b| format!("{}\n{}", a, b))
            .into_bytes()
    });

    let mut stream = controller.calculate_primes(
        prime_opts.start_offset,
        prime_opts.max_number,
        prime_opts.numbers_per_step,
        prime_opts.local_size.unwrap_or(128),
        !prime_opts.no_cache,
    );
    while let Ok(r) = stream.next() {
        let primes = r.value();
        if prime_opts.cpu_validate {
            validate_primes_on_cpu(primes);
        }
        let first = *primes.first().unwrap(); // if there's none, rip
        log::debug!(
            "Calculated {} primes in {:?}, offset: {}",
            primes.len(),
            r.gpu_duration(),
            first
        );
        csv_writer.add_row(vec![
            Local::now().format("%Y-%m-%dT%H:%M:%S.%f").to_string(),
            first.to_string(),
            primes.len().to_string(),
            duration_to_ms_string(r.gpu_duration()),
        ]);
        output_writer.write(primes.clone());
    }
    csv_writer.close();
    output_writer.close();

    Ok(())
}

/// Benchmarks the local size used for calculations
fn bench_local_size(opts: BenchLocalSize, mut controller: KernelController) -> OCLStreamResult<()> {
    set_output_colored(opts.bench_options.general_options.color);
    controller.set_concurrency(opts.bench_options.general_options.threads);
    let bench_output = opts
        .bench_options
        .benchmark_file
        .unwrap_or(PathBuf::from(format!(
            "bench_local_{}-{}-{}_g{}_r{}_s{}_{}.csv",
            opts.local_size_start,
            opts.local_size_step,
            opts.local_size_stop,
            opts.global_size,
            opts.bench_options.repetitions,
            opts.bench_options.calculation_steps,
            Local::now().format("%Y%m%d%H%M%S")
        )));
    let bench_writer = open_write_buffered(&bench_output);
    let csv_writer = ThreadedCSVWriter::new(
        bench_writer,
        &[
            "timestamp",
            "local_size",
            "global_size",
            "calc_count",
            "write_duration",
            "gpu_duration",
            "read_duration",
        ],
    );
    let stream = controller.bench_local_size(
        opts.global_size,
        opts.local_size_start,
        opts.local_size_step,
        opts.local_size_stop,
        opts.bench_options.calculation_steps,
        opts.bench_options.repetitions,
    )?;
    read_bench_results(opts.bench_options.calculation_steps, csv_writer, stream);

    Ok(())
}

/// Benchmarks the global size used for calculations
fn bench_global_size(
    opts: BenchGlobalSize,
    mut controller: KernelController,
) -> OCLStreamResult<()> {
    set_output_colored(opts.bench_options.general_options.color);
    controller.set_concurrency(opts.bench_options.general_options.threads);
    let bench_output = opts
        .bench_options
        .benchmark_file
        .unwrap_or(PathBuf::from(format!(
            "bench_global_{}-{}-{}_l{}_r{}_s{}_{}.csv",
            opts.global_size_start,
            opts.global_size_step,
            opts.global_size_stop,
            opts.local_size,
            opts.bench_options.repetitions,
            opts.bench_options.calculation_steps,
            Local::now().format("%Y%m%d%H%M%S")
        )));
    let bench_writer = open_write_buffered(&bench_output);
    let csv_writer = ThreadedCSVWriter::new(
        bench_writer,
        &[
            "timestamp",
            "local_size",
            "global_size",
            "calc_count",
            "write_duration",
            "gpu_duration",
            "read_duration",
        ],
    );
    let stream = controller.bench_global_size(
        opts.local_size,
        opts.global_size_start,
        opts.global_size_step,
        opts.global_size_stop,
        opts.bench_options.calculation_steps,
        opts.bench_options.repetitions,
    )?;
    read_bench_results(opts.bench_options.calculation_steps, csv_writer, stream);

    Ok(())
}

/// Reads benchmark results from the stream and prints
/// them to the console
fn read_bench_results(
    calculation_steps: u32,
    mut csv_writer: ThreadedCSVWriter,
    mut stream: OCLStream<BenchStatistics>,
) {
    loop {
        match stream.next() {
            Ok(stats) => {
                log::debug!("{:?}", stats);
                csv_writer.add_row(vec![
                    Local::now().format("%Y-%m-%dT%H:%M:%S.%f").to_string(),
                    stats.local_size.to_string(),
                    stats.global_size.to_string(),
                    calculation_steps.to_string(),
                    duration_to_ms_string(&stats.write_duration),
                    duration_to_ms_string(&stats.calc_duration),
                    duration_to_ms_string(&stats.read_duration),
                ])
            }
            _ => {
                break;
            }
        }
    }
    csv_writer.close();
}

fn validate_primes_on_cpu(primes: &Vec<u64>) {
    log::debug!("Validating primes on the cpu");
    let failures = primes
        .par_iter()
        .filter(|n| !is_prime(**n))
        .collect::<Vec<&u64>>();
    if failures.len() > 0 {
        panic!(
            "{} failures in prime calculation: {:?}",
            failures.len(),
            failures
        );
    } else {
        log::debug!("No failures found.");
    }
}

fn duration_to_ms_string(duration: &Duration) -> String {
    format!("{}", duration.as_secs_f64() * 1000f64)
}

/// opens a file in a buffered writer
/// if it already exists it will be recreated
fn open_write_buffered(path: &PathBuf) -> BufWriter<File> {
    BufWriter::new(
        OpenOptions::new()
            .truncate(true)
            .write(true)
            .create(true)
            .open(path)
            .expect("Failed to open file!"),
    )
}

fn set_output_colored(colored: UseColors) {
    match colored {
        UseColors::On => colored::control::set_override(true),
        UseColors::Off => colored::control::set_override(false),
        _ => {}
    }
}

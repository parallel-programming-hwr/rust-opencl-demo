/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */

use crate::kernel_controller::primes::is_prime;
use crate::kernel_controller::KernelController;
use rayon::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::mem;
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use structopt::StructOpt;

mod kernel_controller;

#[derive(StructOpt, Clone, Debug)]
#[structopt()]
enum Opts {
    /// Calculates primes on the GPU
    #[structopt(name = "calculate-primes")]
    CalculatePrimes(CalculatePrimes),
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

fn main() -> ocl::Result<()> {
    let opts: Opts = Opts::from_args();
    let controller = KernelController::new()?;

    match opts {
        Opts::CalculatePrimes(prime_opts) => calculate_primes(prime_opts, controller),
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
    let mut timings = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(prime_opts.timings_file)
            .unwrap(),
    );
    timings
        .write_all("offset,count,gpu_duration,filter_duration,duration\n".as_bytes())
        .unwrap();
    let (sender, handle) = create_write_thread(output);

    let mut offset = prime_opts.start_offset;
    if offset % 2 == 0 {
        offset += 1;
    }
    if offset < 2 {
        sender.send(vec![2]).unwrap();
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
        timings
            .write_all(
                format!(
                    "{},{},{},{},{}\n",
                    offset,
                    primes.len(),
                    prime_result.gpu_duration.as_secs_f64() * 1000f64,
                    prime_result.filter_duration.as_secs_f64() * 1000f64,
                    elapsed_ms
                )
                .as_bytes(),
            )
            .unwrap();
        timings.flush().unwrap();

        if prime_opts.cpu_validate {
            validate_primes_on_cpu(&primes)
        }
        println!();
        sender.send(primes).unwrap();

        if (prime_opts.numbers_per_step as u128 * 2 + offset as u128)
            > prime_opts.max_number as u128
        {
            break;
        }
        offset += prime_opts.numbers_per_step as u64 * 2;
    }

    mem::drop(sender);
    handle.join().unwrap();

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

fn create_write_thread(mut writer: BufWriter<File>) -> (Sender<Vec<u64>>, JoinHandle<()>) {
    let (tx, rx) = channel();
    let handle = thread::spawn(move || {
        for primes in rx {
            for prime in primes {
                writer.write_all(format!("{}\n", prime).as_bytes()).unwrap();
            }
            writer.flush().unwrap();
        }
    });

    (tx, handle)
}

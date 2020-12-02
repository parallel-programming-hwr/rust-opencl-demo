/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */

use crate::kernel_controller::KernelController;
use parking_lot::Mutex;
use std::mem::size_of;
use std::sync::Arc;
use std::time::{Duration, Instant};

pub struct PrimeCalculationResult {
    pub primes: Vec<u64>,
    pub gpu_duration: Duration,
    pub filter_duration: Duration,
}

impl KernelController {
    /// Filters all primes from the input without using a precalculated list of primes
    /// for divisibility checks
    pub fn filter_primes_simple(&self, input: Vec<u64>) -> ocl::Result<PrimeCalculationResult> {
        let input_buffer = self.pro_que.buffer_builder().len(input.len()).build()?;
        input_buffer.write(&input[..]).enq()?;

        let output_buffer = self
            .pro_que
            .buffer_builder()
            .len(input.len())
            .fill_val(0u8)
            .build()?;

        let kernel = self
            .pro_que
            .kernel_builder("check_prime")
            .arg(&input_buffer)
            .arg(&output_buffer)
            .global_work_size(input.len())
            .build()?;

        let start = Instant::now();
        unsafe {
            kernel.enq()?;
        }

        self.pro_que.finish()?;
        let gpu_calc_duration = start.elapsed();

        let mut output = vec![0u8; output_buffer.len()];
        output_buffer.read(&mut output).enq()?;
        println!(
            "GPU IO + Calculation took {} ms",
            gpu_calc_duration.as_secs_f64() * 1000f64
        );

        let filter_start = Instant::now();
        let primes = map_gpu_prime_result(input, output);

        Ok(PrimeCalculationResult {
            primes,
            filter_duration: filter_start.elapsed(),
            gpu_duration: gpu_calc_duration,
        })
    }

    /// Filters the primes from a list of numbers by using a precalculated list of primes to check
    /// for divisibility
    pub fn filter_primes(&self, input: Vec<u64>) -> ocl::Result<PrimeCalculationResult> {
        lazy_static::lazy_static! {static ref PRIME_CACHE: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));}
        if PRIME_CACHE.lock().len() == 0 {
            PRIME_CACHE.lock().append(&mut get_primes(
                (*input.iter().max().unwrap_or(&1024) as f64).sqrt().ceil() as u64,
            ));
        }

        let prime_buffer = self
            .pro_que
            .buffer_builder()
            .len(PRIME_CACHE.lock().len())
            .build()?;

        prime_buffer.write(&PRIME_CACHE.lock()[..]).enq()?;

        let input_buffer = self.pro_que.buffer_builder().len(input.len()).build()?;
        input_buffer.write(&input[..]).enq()?;

        let output_buffer = self
            .pro_que
            .buffer_builder()
            .len(input.len())
            .fill_val(0u8)
            .build()?;

        let kernel = self
            .pro_que
            .kernel_builder("check_prime_cached")
            .arg(prime_buffer.len() as u32)
            .arg(&prime_buffer)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .global_work_size(input.len())
            .build()?;

        let start = Instant::now();
        unsafe {
            kernel.enq()?;
        }

        self.pro_que.finish()?;
        let gpu_calc_duration = start.elapsed();

        let mut output = vec![0u8; output_buffer.len()];
        output_buffer.read(&mut output).enq()?;

        println!(
            "GPU IO + Calculation took {} ms",
            gpu_calc_duration.as_secs_f64() * 1000f64
        );

        let prime_filter_start = Instant::now();
        let primes = map_gpu_prime_result(input, output);
        let filter_duration = prime_filter_start.elapsed();

        let prime_calc_start = Instant::now();
        let mut prime_cache = PRIME_CACHE.lock();

        if (prime_cache.len() + primes.len()) * size_of::<i64>()
            < self.available_memory()? as usize / 4
        {
            prime_cache.append(&mut primes.clone());
            prime_cache.sort();
            prime_cache.dedup();
        }
        let cache_duration = prime_calc_start.elapsed();
        println!(
            "Prime caching took: {} ms, size: {}",
            cache_duration.as_secs_f64() * 1000f64,
            prime_cache.len(),
        );

        Ok(PrimeCalculationResult {
            primes,
            gpu_duration: gpu_calc_duration,
            filter_duration,
        })
    }
}

/// Returns a list of prime numbers that can be used to speed up the divisibility check
fn get_primes(max_number: u64) -> Vec<u64> {
    let start = Instant::now();
    let mut primes = Vec::with_capacity((max_number as f64).sqrt() as usize);
    let mut num = 1;

    while num < max_number {
        let mut is_prime = true;

        if num == 2 || num == 3 {
            is_prime = true;
        } else if num == 1 || num % 2 == 0 {
            is_prime = false;
        } else {
            let check_stop = (num as f64).sqrt().ceil() as u64;

            if check_stop <= 9 {
                for i in (3..check_stop).step_by(2) {
                    if num % i == 0 {
                        is_prime = false;
                    }
                }
            } else {
                for i in (9..(check_stop + 6)).step_by(6) {
                    if num % (i - 2) == 0 || num % (i - 4) == 0 {
                        is_prime = false;
                    }
                }
            }
        }
        if is_prime {
            primes.push(num)
        }
        num += 2;
    }
    println!(
        "Generated {} primes on the cpu in {} ms",
        primes.len(),
        start.elapsed().as_secs_f64() * 1000f64,
    );

    primes
}

pub fn is_prime(number: u64) -> bool {
    if number == 2 || number == 3 {
        return true;
    }
    if number == 1 || number % 2 == 0 {
        return false;
    }
    let limit = (number as f64).sqrt().ceil() as u64;
    for i in (3..limit).step_by(2) {
        if number % i == 0 {
            return false;
        }
    }

    return true;
}

#[inline]
fn map_gpu_prime_result(input: Vec<u64>, output: Vec<u8>) -> Vec<u64> {
    input
        .into_iter()
        .enumerate()
        .filter(|(index, _)| output[*index] == 1)
        .map(|(_, v)| v)
        .collect::<Vec<u64>>()
}

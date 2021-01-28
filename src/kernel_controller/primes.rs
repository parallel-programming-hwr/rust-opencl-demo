/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crate::benching::enqueue_profiled;
use crate::benching::result::ProfiledResult;
use crate::kernel_controller::KernelController;
use crate::utils::progress::get_progress_bar;
use ocl::ProQue;
use ocl_stream::stream::OCLStream;
use ocl_stream::traits::ToOclBuffer;
use parking_lot::Mutex;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;
use std_semaphore::Semaphore;

const MEMORY_LIMIT: u64 = 4 * 1024 * 1024 * 1024;

impl KernelController {
    /// Calculates prime number on the cpu
    pub fn calculate_primes_cpu(
        &mut self,
        mut start: u64,
        stop: u64,
        step: usize,
    ) -> OCLStream<ProfiledResult<Vec<u64>>> {
        if start % 2 == 0 {
            start += 1;
        }
        log::debug!(
            "Calculating primes between {} and {} with {} number per step on the cpu",
            start,
            stop,
            step,
        );
        let offset = Arc::new(AtomicU64::new(start));
        let pb = get_progress_bar((stop - start) / (step * 2) as u64);

        self.executor.execute_bounded(step * 10, move |ctx| {
            loop {
                if offset.load(Ordering::SeqCst) >= stop {
                    log::trace!("Stop reached.");
                    break;
                }
                let offset = offset.fetch_add(step as u64 * 2, Ordering::SeqCst);
                log::trace!("Calculating {} primes beginning from {}", step, offset);
                let start = Instant::now();

                let primes = (offset..(step as u64 * 2 + offset))
                    .step_by(2)
                    .filter(|n| is_prime(*n))
                    .collect::<Vec<u64>>();

                ctx.sender()
                    .send(ProfiledResult::new(start.elapsed(), primes))?;
                pb.inc(1);
            }

            Ok(())
        })
    }

    /// Calculates prime numbers on the gpu
    pub fn calculate_primes(
        &self,
        mut start: u64,
        stop: u64,
        step: usize,
        local_size: usize,
        use_cache: bool,
    ) -> OCLStream<ProfiledResult<Vec<u64>>> {
        if start % 2 == 0 {
            start += 1;
        }
        log::debug!(
            "Calculating primes between {} and {} with {} number per step and a local size of {}",
            start,
            stop,
            step,
            local_size
        );
        let offset = Arc::new(AtomicU64::new(start));
        let prime_cache = Arc::new(Mutex::new(Vec::new()));

        if use_cache {
            prime_cache
                .lock()
                .append(&mut get_primes(start + step as u64));
        }

        let pb = get_progress_bar((stop - start) / (step * 2) as u64);
        let sem = Semaphore::new(1);

        self.executor.execute_bounded(step * 10, move |ctx| {
            loop {
                let pro_que = ctx.pro_que();
                let sender = ctx.sender();
                if offset.load(Ordering::SeqCst) >= stop {
                    log::trace!("Stop reached.");
                    break;
                }
                let offset = offset.fetch_add(step as u64 * 2, Ordering::SeqCst);
                log::trace!("Calculating {} primes beginning from {}", step, offset);

                let numbers = (offset..(step as u64 * 2 + offset))
                    .step_by(2)
                    .collect::<Vec<u64>>();
                let result = if use_cache {
                    let prime_cache = Arc::clone(&prime_cache);
                    log::trace!("Using optimized function with cached primes");
                    Self::filter_primes_cached(pro_que, numbers, local_size, prime_cache, &sem)?
                } else {
                    log::trace!("Using normal prime calculation function");
                    Self::filter_primes(pro_que, numbers, local_size, &sem)?
                };
                sender.send(result)?;
                pb.inc(1);
            }

            Ok(())
        })
    }

    /// Creates the prime filter kernel and executes it
    fn filter_primes(
        pro_que: &ProQue,
        numbers: Vec<u64>,
        local_size: usize,
        sem: &Semaphore,
    ) -> ocl::Result<ProfiledResult<Vec<u64>>> {
        sem.acquire();
        log::trace!("Creating 0u8 output buffer");
        let output_buffer = pro_que
            .buffer_builder()
            .len(numbers.len())
            .fill_val(0u8)
            .build()?;
        sem.release();

        sem.acquire();
        let input_buffer = numbers.to_ocl_buffer(pro_que)?;
        sem.release();

        sem.acquire();
        log::trace!("Building 'check_prime' kernel");
        let kernel = pro_que
            .kernel_builder("check_prime")
            .local_work_size(local_size)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .global_work_size(numbers.len())
            .build()?;
        sem.release();
        let duration = enqueue_profiled(pro_que, &kernel, &sem)?;

        log::trace!("Reading output");
        let mut output = vec![0u8; output_buffer.len()];
        sem.acquire();
        output_buffer.read(&mut output).enq()?;
        sem.release();

        log::trace!("Filtering primes");
        let primes = map_gpu_prime_result(numbers, output);
        log::trace!("Calculated {} primes", primes.len());

        Ok(ProfiledResult::new(duration, primes))
    }

    /// Filters primes by using the primes from previous
    /// calculations for divisibility checks
    pub fn filter_primes_cached(
        pro_que: &ProQue,
        numbers: Vec<u64>,
        local_size: usize,
        prime_cache: Arc<Mutex<Vec<u64>>>,
        sem: &Semaphore,
    ) -> ocl::Result<ProfiledResult<Vec<u64>>> {
        sem.acquire();
        let prime_buffer = prime_cache.lock().to_ocl_buffer(pro_que)?;
        sem.release();

        sem.acquire();
        let input_buffer = numbers.to_ocl_buffer(pro_que)?;
        sem.release();

        sem.acquire();
        log::trace!("Creating output buffer");
        let output_buffer = pro_que
            .buffer_builder()
            .len(numbers.len())
            .fill_val(0u8)
            .build()?;
        sem.release();

        log::trace!("Building 'check_prime_cached' kernel");
        sem.acquire();
        let kernel = pro_que
            .kernel_builder("check_prime_cached")
            .local_work_size(local_size)
            .arg(prime_buffer.len() as u32)
            .arg(&prime_buffer)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .global_work_size(numbers.len())
            .build()?;
        sem.release();

        let duration = enqueue_profiled(pro_que, &kernel, sem)?;

        log::trace!("Reading output");
        let mut output = vec![0u8; output_buffer.len()];
        sem.acquire();
        output_buffer.read(&mut output).enq()?;
        sem.release();

        log::trace!("Mapping prime result");
        let primes = map_gpu_prime_result(numbers, output);
        log::trace!("Calculated {} primes", primes.len());

        let mut prime_cache = prime_cache.lock();

        log::trace!("Updating prime cache");
        if (prime_cache.len() + primes.len()) * size_of::<i64>() < MEMORY_LIMIT as usize / 4 {
            prime_cache.append(&mut primes.clone());
            prime_cache.sort();
            prime_cache.dedup();
        }

        Ok(ProfiledResult::new(duration, primes))
    }
}

/// Returns a list of prime numbers that can be used to speed up the divisibility check
fn get_primes(max_number: u64) -> Vec<u64> {
    log::trace!("Calculating primes until {} on the cpu", max_number);
    let start = Instant::now();
    let mut primes = Vec::with_capacity((max_number as f64).sqrt() as usize);
    let mut num = 1;

    while num < max_number {
        let is_prime = is_prime(num);

        if is_prime {
            primes.push(num)
        }
        num += 2;
    }
    log::trace!(
        "Generated {} primes on the cpu in {} ms",
        primes.len(),
        start.elapsed().as_secs_f64() * 1000f64,
    );

    primes
}

/// Checks if a given number is a prime number
pub(crate) fn is_prime(num: u64) -> bool {
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
    is_prime
}

#[inline]
pub fn map_gpu_prime_result(input: Vec<u64>, output: Vec<u8>) -> Vec<u64> {
    input
        .into_iter()
        .enumerate()
        .filter(|(index, _)| output[*index] == 1)
        .map(|(_, v)| v)
        .collect::<Vec<u64>>()
}

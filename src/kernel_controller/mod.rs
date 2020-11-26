/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */

use ocl::core::DeviceInfo;
use ocl::enums::DeviceInfoResult;
use ocl::ProQue;
use parking_lot::Mutex;
use std::cmp::max;
use std::mem::size_of;
use std::sync::Arc;
use std::time::Instant;

pub struct KernelController {
    pro_que: ProQue,
}

impl KernelController {
    pub fn new() -> ocl::Result<Self> {
        let pro_que = ProQue::builder()
            .src(include_str!("kernel.cl"))
            .dims(1 << 20)
            .build()?;
        let device = pro_que.device();
        println!("Using device {}", device.name()?);
        println!("Vendor: {}", device.vendor()?);
        println!(
            "Global Mem Size: {} bytes",
            device.info(DeviceInfo::GlobalMemSize)?
        );
        println!(
            "Max Mem Alloc: {} bytes",
            device.info(DeviceInfo::MaxMemAllocSize)?
        );
        println!();
        Ok(Self { pro_que })
    }

    fn available_memory(&self) -> ocl::Result<u64> {
        match self.pro_que.device().info(DeviceInfo::GlobalMemSize)? {
            DeviceInfoResult::GlobalMemSize(size) => Ok(size),
            _ => Ok(0),
        }
    }

    pub fn filter_primes(&self, input: Vec<u64>) -> ocl::Result<Vec<u64>> {
        lazy_static::lazy_static! {static ref PRIME_CACHE: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(get_primes(2048)));}

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
            .kernel_builder("check_prime")
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

        let mut output = vec![0u8; output_buffer.len()];
        output_buffer.read(&mut output).enq()?;

        let mut input_o = vec![0u64; input_buffer.len()];
        input_buffer.read(&mut input_o).enq()?;

        println!(
            "GPU IO + Calculation took {} ms",
            start.elapsed().as_secs_f64() * 1000f64
        );

        let primes = input
            .iter()
            .enumerate()
            .filter(|(index, _)| output[*index] == 1)
            .map(|(_, v)| *v)
            .collect::<Vec<u64>>();

        let start = Instant::now();
        let mut prime_cache = PRIME_CACHE.lock();

        if (prime_cache.len() + primes.len()) * size_of::<i64>()
            < self.available_memory()? as usize / 4
        {
            prime_cache.append(&mut primes.clone());
            prime_cache.sort();
            prime_cache.dedup();
        }
        println!(
            "Prime caching took: {} ms, size: {}",
            start.elapsed().as_secs_f64() * 1000f64,
            prime_cache.len(),
        );

        Ok(primes)
    }
}

/// Returns a list of prime numbers that can be used to speed up the divisibility check
fn get_primes(count: usize) -> Vec<u64> {
    let start = Instant::now();
    let mut primes = Vec::with_capacity(count);
    let mut num = 3;

    while primes.len() < count {
        let mut is_prime = true;

        if num < 3 || num % 2 == 0 {
            is_prime = false;
        } else {
            let check_stop = (num as f64).sqrt().ceil() as u64;
            let mut free_check_start = 9;

            for prime in primes.iter().take_while(|num| **num < check_stop) {
                let prime = *prime;
                free_check_start = prime;
                if num % prime == 0 {
                    is_prime = false;
                    break;
                }
            }
            if free_check_start < check_stop && is_prime {
                free_check_start -= free_check_start % 3;
                if free_check_start % 2 == 0 {
                    free_check_start -= 3;
                }
                for i in (max(free_check_start, 9)..check_stop).step_by(6) {
                    if num % (i - 2) == 0 || num % (i - 4) == 0 {
                        is_prime = false;
                        break;
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
        count,
        start.elapsed().as_secs_f64() * 1000f64
    );

    primes
}

#[allow(dead_code)]
fn is_prime(number: u64) -> bool {
    if number < 3 || number % 2 == 0 {
        return false;
    }
    for i in (9..(number as f64).sqrt().ceil() as u64).step_by(6) {
        if number % (i - 2) == 0 || number % (i - 4) == 0 {
            return false;
        }
    }
    return true;
}

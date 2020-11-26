/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */

use ocl::ProQue;
use parking_lot::Mutex;
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
        Ok(Self { pro_que })
    }

    pub fn filter_primes(&self, input: Vec<i64>) -> ocl::Result<Vec<i64>> {
        lazy_static::lazy_static! {static ref PRIME_CACHE: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(get_lower_primes(2048)));}

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
            .arg(prime_buffer.len() as i32)
            .arg(&prime_buffer)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .global_work_size(input.len())
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        let mut output = vec![0u8; output_buffer.len()];
        output_buffer.read(&mut output).enq()?;

        let mut input_o = vec![0i64; input_buffer.len()];
        input_buffer.read(&mut input_o).enq()?;

        let primes = input
            .iter()
            .enumerate()
            .filter(|(index, _)| output[*index] == 1)
            .map(|(_, v)| *v)
            .collect::<Vec<i64>>();

        let start = Instant::now();
        let mut prime_cache = PRIME_CACHE.lock();

        if prime_cache.len() < 1024 * 1024 * 1024 {
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
fn get_lower_primes(count: usize) -> Vec<i64> {
    let mut primes = Vec::new();
    let mut num = 3;

    while primes.len() < count {
        let mut is_prime = true;

        if num < 3 || num % 2 == 0 {
            is_prime = false;
        } else {
            for i in (3..((num as f64).sqrt().ceil() as i64)).step_by(2) {
                if num % i == 0 {
                    is_prime = false;
                    break;
                }
            }
        }
        if is_prime {
            primes.push(num)
        }
        num += 2;
    }

    primes
}

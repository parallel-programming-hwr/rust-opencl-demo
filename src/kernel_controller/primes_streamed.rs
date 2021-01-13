use crate::benching::enqueue_profiled;
use crate::benching::result::ProfiledResult;
use crate::kernel_controller::primes::{get_primes, map_gpu_prime_result};
use crate::kernel_controller::KernelController;
use ocl::ProQue;
use ocl_stream::stream::OCLStream;
use ocl_stream::traits::ToOclBuffer;
use parking_lot::Mutex;
use std::mem::size_of;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const MEMORY_LIMIT: u64 = 4 * 1024 * 1024 * 1024;

impl KernelController {
    pub fn get_primes(
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
        let offset = Arc::new(AtomicU64::new(start));
        let prime_cache = Arc::new(Mutex::new(Vec::new()));
        if use_cache {
            prime_cache
                .lock()
                .append(&mut get_primes(start + step as u64));
        }

        self.executor.execute_bounded(step * 10, move |ctx| {
            loop {
                let pro_que = ctx.pro_que();
                let sender = ctx.sender();
                if offset.load(Ordering::SeqCst) >= stop {
                    break;
                }
                let offset = offset.fetch_add(step as u64 * 2, Ordering::SeqCst);
                let numbers = (offset..(step as u64 * 2 + offset))
                    .step_by(2)
                    .collect::<Vec<u64>>();
                let result = if use_cache {
                    let prime_cache = Arc::clone(&prime_cache);
                    Self::filter_primes_streamed_cached(pro_que, numbers, local_size, prime_cache)?
                } else {
                    Self::filter_primes_streamed(pro_que, numbers, local_size)?
                };
                sender.send(result)?;
            }

            Ok(())
        })
    }

    /// Creates the prime filter kernel and executes it
    fn filter_primes_streamed(
        pro_que: &ProQue,
        numbers: Vec<u64>,
        local_size: usize,
    ) -> ocl::Result<ProfiledResult<Vec<u64>>> {
        let output_buffer = pro_que
            .buffer_builder()
            .len(numbers.len())
            .fill_val(0u8)
            .build()?;
        let input_buffer = numbers.to_ocl_buffer(pro_que)?;

        let kernel = pro_que
            .kernel_builder("check_prime")
            .local_work_size(local_size)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .global_work_size(numbers.len())
            .build()?;
        let duration = enqueue_profiled(pro_que, &kernel)?;

        let mut output = vec![0u8; output_buffer.len()];
        output_buffer.read(&mut output).enq()?;
        let primes = map_gpu_prime_result(numbers, output);

        Ok(ProfiledResult::new(duration, primes))
    }

    pub fn filter_primes_streamed_cached(
        pro_que: &ProQue,
        numbers: Vec<u64>,
        local_size: usize,
        prime_cache: Arc<Mutex<Vec<u64>>>,
    ) -> ocl::Result<ProfiledResult<Vec<u64>>> {
        let prime_buffer = prime_cache.lock().to_ocl_buffer(pro_que)?;
        let input_buffer = numbers.to_ocl_buffer(pro_que)?;

        let output_buffer = pro_que
            .buffer_builder()
            .len(numbers.len())
            .fill_val(0u8)
            .build()?;

        let kernel = pro_que
            .kernel_builder("check_prime_cached")
            .local_work_size(local_size)
            .arg(prime_buffer.len() as u32)
            .arg(&prime_buffer)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .global_work_size(numbers.len())
            .build()?;

        let duration = enqueue_profiled(pro_que, &kernel)?;

        let mut output = vec![0u8; output_buffer.len()];
        output_buffer.read(&mut output).enq()?;

        let primes = map_gpu_prime_result(numbers, output);

        let mut prime_cache = prime_cache.lock();

        if (prime_cache.len() + primes.len()) * size_of::<i64>() < MEMORY_LIMIT as usize / 4 {
            prime_cache.append(&mut primes.clone());
            prime_cache.sort();
            prime_cache.dedup();
        }

        Ok(ProfiledResult::new(duration, primes))
    }
}

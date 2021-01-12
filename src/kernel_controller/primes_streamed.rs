use crate::benching::enqueue_profiled;
use crate::benching::result::ProfiledResult;
use crate::kernel_controller::primes::map_gpu_prime_result;
use crate::kernel_controller::KernelController;
use ocl::ProQue;
use ocl_stream::stream::OCLStream;
use ocl_stream::traits::ToOclBuffer;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

impl KernelController {
    pub fn get_primes(
        &self,
        mut start: u64,
        stop: u64,
        step: usize,
        local_size: usize,
    ) -> OCLStream<ProfiledResult<Vec<u64>>> {
        if start % 2 == 0 {
            start += 1;
        }
        let offset = Arc::new(AtomicU64::new(start));
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
                let result = Self::filter_primes_streamed(pro_que, numbers, local_size)?;
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
}

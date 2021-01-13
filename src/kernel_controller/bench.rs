/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use std::fmt::{self, Display, Formatter};
use std::ops::Deref;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use ocl_stream::executor::context::ExecutorContext;
use ocl_stream::executor::stream::OCLStream;
use ocl_stream::traits::*;
use ocl_stream::utils::result::OCLStreamResult;
use ocl_stream::utils::shared_buffer::SharedBuffer;

use crate::benching::enqueue_profiled;
use crate::kernel_controller::KernelController;
use crate::utils::progress::get_progress_bar;

#[derive(Clone, Debug)]
pub struct BenchStatistics {
    pub calc_count: u32,
    pub global_size: usize,
    pub local_size: usize,
    pub write_duration: Duration,
    pub calc_duration: Duration,
    pub read_duration: Duration,
}

impl Display for BenchStatistics {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Calculation Count: {}\nTask Count: {}\nLocal Size: {}\nWrite Duration: {} ms\nGPU Duration: {} ms\nRead Duration: {} ms",
            self.calc_count,
            self.global_size,
            self.local_size,
            self.write_duration.as_secs_f64() * 1000f64,
            self.calc_duration.as_secs_f64() * 1000f64,
            self.read_duration.as_secs_f64() * 1000f64
        )
    }
}

impl KernelController {
    /// Benchmarks the value for the global size
    pub fn bench_global_size(
        &self,
        local_size: usize,
        global_size_start: usize,
        global_size_step: usize,
        global_size_stop: usize,
        calc_count: u32,
        repetitions: usize,
    ) -> OCLStreamResult<OCLStream<BenchStatistics>> {
        log::debug!("Benchmarking global size. Global Size: {}, Start: {}, Step: {} ,Stop: {}, Calculations: {}, Repetitions: {}",
                    local_size, global_size_start, global_size_step, global_size_stop, calc_count, repetitions);
        let global_size = AtomicUsize::new(global_size_start);
        let pb = get_progress_bar(
            ((global_size_stop - global_size_start) / global_size_step) as u64 * repetitions as u64,
        );

        let stream = self.executor.execute_bounded(global_size_stop, move |ctx| {
            loop {
                if global_size.load(Ordering::SeqCst) > global_size_stop {
                    log::trace!("Stop reached");
                    break;
                }
                let global_size = global_size.fetch_add(global_size_step, Ordering::SeqCst);

                if global_size % local_size != 0 {
                    log::trace!("Global size not divisible by local size. Continuing");
                    pb.inc(repetitions as u64);
                    continue;
                }
                let input_buffer: SharedBuffer<u32> =
                    vec![0u32; global_size].to_shared_buffer(ctx.pro_que())?;

                log::trace!(
                    "Benching global size {} with {} repetitions",
                    global_size,
                    repetitions
                );
                for _ in 0..repetitions {
                    let stats =
                        Self::bench_int(&ctx, local_size, calc_count, input_buffer.clone())?;
                    ctx.sender().send(stats)?;
                    pb.inc(1);
                }
            }
            Ok(())
        });

        Ok(stream)
    }

    /// Benchmarks the value for the local size
    pub fn bench_local_size(
        &self,
        global_size: usize,
        local_size_start: usize,
        local_size_step: usize,
        local_size_stop: usize,
        calc_count: u32,
        repetitions: usize,
    ) -> OCLStreamResult<OCLStream<BenchStatistics>> {
        log::debug!("Benchmarking local size. Global Size: {}, Start: {}, Step: {} ,Stop: {}, Calculations: {}, Repetitions: {}",
                    global_size, local_size_start, local_size_step, local_size_stop, calc_count, repetitions);

        let input_buffer: SharedBuffer<u32> =
            vec![0u32; global_size].to_shared_buffer(self.executor.pro_que())?;
        let local_size = AtomicUsize::new(local_size_start);
        let pb = get_progress_bar(
            ((local_size_stop - local_size_start) / local_size_step) as u64 * repetitions as u64,
        );

        let stream = self.executor.execute_bounded(global_size, move |ctx| {
            loop {
                if local_size.load(Ordering::SeqCst) > local_size_stop {
                    log::trace!("Stop Reached");
                    break;
                }
                let local_size = local_size.fetch_add(local_size_step, Ordering::SeqCst);

                if local_size > 1024 || global_size % local_size != 0 {
                    log::trace!("Global size not divisible by local size. Continuing");
                    pb.inc(repetitions as u64);
                    continue;
                }

                log::trace!(
                    "Benching local size {} with {} repetitions",
                    local_size,
                    repetitions
                );
                for _ in 0..repetitions {
                    let stats =
                        Self::bench_int(&ctx, local_size, calc_count, input_buffer.clone())?;
                    ctx.sender().send(stats)?;
                    pb.inc(1);
                }
            }
            Ok(())
        });

        Ok(stream)
    }

    /// Benches an integer
    fn bench_int(
        ctx: &ExecutorContext<BenchStatistics>,
        local_size: usize,
        calc_count: u32,
        input_buffer: SharedBuffer<u32>,
    ) -> ocl::Result<BenchStatistics> {
        let num_tasks = input_buffer.inner().lock().len();

        log::trace!("Building kernel");
        let kernel = ctx
            .pro_que()
            .kernel_builder("bench_int")
            .local_work_size(local_size)
            .global_work_size(num_tasks)
            .arg(calc_count)
            .arg(input_buffer.inner().lock().deref())
            .build()?;

        let calc_duration = enqueue_profiled(ctx.pro_que(), &kernel)?;

        log::trace!("Reading output");
        let mut output = vec![0u32; num_tasks];
        let read_start = Instant::now();
        input_buffer.read(&mut output)?;
        let read_duration = read_start.elapsed();

        Ok(BenchStatistics {
            global_size: num_tasks,
            calc_count,
            local_size,
            read_duration,
            calc_duration,
            write_duration: Duration::from_nanos(0),
        })
    }
}

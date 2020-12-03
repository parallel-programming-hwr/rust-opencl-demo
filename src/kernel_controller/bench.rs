/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */

use crate::kernel_controller::KernelController;
use std::time::{Duration, Instant};

pub struct BenchStatistics {
    pub calc_count: u32,
    pub num_tasks: usize,
    pub write_duration: Duration,
    pub calc_duration: Duration,
    pub read_duration: Duration,
}

impl KernelController {
    /// Benches an integer
    pub fn bench_int(&self, calc_count: u32, num_tasks: usize) -> ocl::Result<BenchStatistics> {
        let write_start = Instant::now();
        let input_buffer = self
            .pro_que
            .buffer_builder()
            .len(num_tasks)
            .fill_val(0u32)
            .build()?;
        let write_duration = write_start.elapsed();

        let kernel = self
            .pro_que
            .kernel_builder("bench_int")
            .arg(calc_count)
            .arg(&input_buffer)
            .build()?;
        let calc_start = Instant::now();
        unsafe {
            kernel.enq()?;
        }
        self.pro_que.finish()?;
        let calc_duration = calc_start.elapsed();
        let mut output = vec![0u32; num_tasks];
        let read_start = Instant::now();
        input_buffer.read(&mut output).enq()?;
        let read_duration = read_start.elapsed();

        Ok(BenchStatistics {
            num_tasks,
            calc_count,
            read_duration,
            calc_duration,
            write_duration,
        })
    }
}

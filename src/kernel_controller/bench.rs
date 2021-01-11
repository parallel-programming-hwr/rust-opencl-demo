/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */

use crate::kernel_controller::KernelController;
use std::fmt::{self, Display, Formatter};
use std::time::{Duration, Instant};

pub struct BenchStatistics {
    pub calc_count: u32,
    pub num_tasks: usize,
    pub local_size: Option<usize>,
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
            self.num_tasks,
            self.local_size.map(|v|v.to_string()).unwrap_or("n/a".to_string()),
            self.write_duration.as_secs_f64() * 1000f64,
            self.calc_duration.as_secs_f64() * 1000f64,
            self.read_duration.as_secs_f64() * 1000f64
        )
    }
}

impl BenchStatistics {
    pub fn avg(&mut self, other: Self) {
        self.read_duration = (self.read_duration + other.read_duration) / 2;
        self.write_duration = (self.write_duration + other.write_duration) / 2;
        self.calc_duration = (self.calc_duration + other.calc_duration) / 2;
    }
}

impl KernelController {
    /// Benches an integer
    pub fn bench_int(
        &self,
        calc_count: u32,
        num_tasks: usize,
        local_size: Option<usize>,
    ) -> ocl::Result<BenchStatistics> {
        let write_start = Instant::now();
        let input_buffer = self
            .pro_que
            .buffer_builder()
            .len(num_tasks)
            .fill_val(0u32)
            .build()?;
        let write_duration = write_start.elapsed();

        let mut builder = self.pro_que.kernel_builder("bench_int");

        if let Some(local_size) = local_size {
            builder.local_work_size(local_size);
        }

        let kernel = builder
            .arg(calc_count)
            .arg(&input_buffer)
            .global_work_size(num_tasks)
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
            local_size,
            read_duration,
            calc_duration,
            write_duration,
        })
    }
}

/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use ocl::core::{get_event_profiling_info, wait_for_event, ProfilingInfo};
use ocl::{EventList, Kernel, ProQue};
use std::time::Duration;

pub mod result;

/// Runs a benchmark on the kernel
/// The ProQue needs to have profiling enabled
pub fn enqueue_profiled(pro_que: &ProQue, kernel: &Kernel) -> ocl::Result<Duration> {
    log::trace!("Running kernel with profiling");
    log::trace!("Enqueueing start event");
    let event_start = pro_que.queue().enqueue_marker::<EventList>(None)?;
    log::trace!("Enqueueing Kernel");

    unsafe {
        kernel.enq()?;
    }
    log::trace!("Enqueueing stop event");
    let event_stop = pro_que.queue().enqueue_marker::<EventList>(None)?;

    log::trace!("Waiting for start event");
    wait_for_event(&event_start)?;
    log::trace!("Waiting for stop event");
    wait_for_event(&event_stop)?;
    let start = get_event_profiling_info(&event_start, ProfilingInfo::End)?;
    let stop = get_event_profiling_info(&event_stop, ProfilingInfo::Start)?;
    let gpu_calc_duration = Duration::from_nanos(stop.time()? - start.time()?);
    log::trace!(
        "Elapsed time between start and stop: {:?}",
        gpu_calc_duration
    );

    Ok(gpu_calc_duration)
}

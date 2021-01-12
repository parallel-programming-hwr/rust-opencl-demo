use ocl::core::{get_event_profiling_info, wait_for_event, ProfilingInfo};
use ocl::{EventList, Kernel, ProQue};
use std::time::Duration;

pub mod result;

/// Runs a benchmark on the kernel
/// The ProQue needs to have profiling enabled
pub fn enqueue_profiled(pro_que: &ProQue, kernel: &Kernel) -> ocl::Result<Duration> {
    let event_start = pro_que.queue().enqueue_marker::<EventList>(None)?;
    unsafe {
        kernel.enq()?;
    }
    let event_stop = pro_que.queue().enqueue_marker::<EventList>(None)?;
    wait_for_event(&event_start)?;
    wait_for_event(&event_stop)?;
    let start = get_event_profiling_info(&event_start, ProfilingInfo::End)?;
    let stop = get_event_profiling_info(&event_stop, ProfilingInfo::Start)?;
    let gpu_calc_duration = Duration::from_nanos(stop.time()? - start.time()?);

    Ok(gpu_calc_duration)
}

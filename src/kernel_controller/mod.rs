/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use ocl::core::DeviceInfo;
use ocl::enums::DeviceInfoResult;
use ocl::{CommandQueueProperties, ProQue};
use ocl_stream::OCLStreamExecutor;

pub mod bench;
pub mod primes;

#[derive(Clone)]
pub struct KernelController {
    pro_que: ProQue,
    executor: OCLStreamExecutor,
}

impl KernelController {
    pub fn new() -> ocl::Result<Self> {
        let pro_que = ProQue::builder()
            .src(include_str!("kernel.cl"))
            .dims(1) // won't be used as buffer sizes are declared explicitly
            .queue_properties(CommandQueueProperties::PROFILING_ENABLE)
            .build()?;
        let mut executor = OCLStreamExecutor::new(pro_que.clone());
        executor.set_concurrency(3);
        println!("Using device {}", pro_que.device().name()?);

        Ok(Self { pro_que, executor })
    }

    /// Sets the amount of executor threads that are used
    pub fn set_concurrency(&mut self, concurrency: usize) {
        self.executor.set_concurrency(concurrency)
    }

    /// Prints information about the gpu capabilities
    pub fn print_info(&self) -> ocl::Result<()> {
        let device = self.pro_que.device();
        let info_keys = vec![
            DeviceInfo::Type,
            DeviceInfo::Vendor,
            DeviceInfo::DriverVersion,
            DeviceInfo::ExecutionCapabilities,
            DeviceInfo::MaxComputeUnits,
            DeviceInfo::MaxWorkGroupSize,
            DeviceInfo::MaxClockFrequency,
            DeviceInfo::GlobalMemSize,
            DeviceInfo::LocalMemSize,
            DeviceInfo::MaxMemAllocSize,
            DeviceInfo::LocalMemType,
            DeviceInfo::GlobalMemCacheType,
            DeviceInfo::GlobalMemCacheSize,
            DeviceInfo::OpenclCVersion,
            DeviceInfo::Platform,
        ];

        for info in info_keys {
            println!("{:?}: {}", info, device.info(info)?)
        }
        println!();

        Ok(())
    }

    #[allow(dead_code)]
    fn available_memory(&self) -> ocl::Result<u64> {
        match self.pro_que.device().info(DeviceInfo::GlobalMemSize)? {
            DeviceInfoResult::GlobalMemSize(size) => Ok(size),
            _ => Ok(0),
        }
    }
}

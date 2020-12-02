/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */

use ocl::core::DeviceInfo;
use ocl::enums::DeviceInfoResult;
use ocl::ProQue;

pub mod primes;

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
}

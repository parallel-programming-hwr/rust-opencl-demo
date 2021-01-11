use crate::kernel_controller::primes::PrimeCalculationResult;
use crate::kernel_controller::KernelController;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Sender;
use std::sync::Arc;
use std::thread::Builder as ThreadBuilder;

pub struct ConcurrentKernelExecutor {
    kernel_controller: KernelController,
}

impl ConcurrentKernelExecutor {
    pub fn new(kernel_controller: KernelController) -> Self {
        Self { kernel_controller }
    }

    pub fn calculate_primes(
        &self,
        mut offset: u64,
        numbers_per_step: usize,
        local_size: Option<usize>,
        stop: u64,
        no_cache: bool,
        num_threads: usize,
        sender: Sender<PrimeCalculationResult>,
    ) {
        let mut handles = Vec::new();
        if offset % 2 == 0 {
            offset += 1;
        }
        let offset = Arc::new(AtomicU64::new(offset));
        let panic = Arc::new(AtomicBool::new(false));

        for i in 0..num_threads {
            let sender = Sender::clone(&sender);
            let controller = self.kernel_controller.clone();
            let offset = Arc::clone(&offset);
            let panic = Arc::clone(&panic);
            let local_size = local_size.clone();

            handles.push(
                ThreadBuilder::new()
                    .name(format!("executor-{}", i))
                    .spawn(move || loop {
                        if panic.load(Ordering::Relaxed) {
                            panic!("Planned panic");
                        }
                        if offset.load(Ordering::SeqCst) >= stop {
                            break;
                        }
                        let offset =
                            offset.fetch_add(numbers_per_step as u64 * 2, Ordering::SeqCst);

                        let numbers = (offset..(numbers_per_step as u64 * 2 + offset))
                            .step_by(2)
                            .collect::<Vec<u64>>();
                        let prime_result = if no_cache {
                            controller
                                .filter_primes_simple(numbers, local_size.clone())
                                .map_err(|e| {
                                    panic.store(true, Ordering::Relaxed);
                                    e
                                })
                                .unwrap()
                        } else {
                            controller
                                .filter_primes(numbers, local_size.clone())
                                .map_err(|e| {
                                    panic.store(true, Ordering::Relaxed);
                                    e
                                })
                                .unwrap()
                        };
                        if let Err(e) = sender.send(prime_result) {
                            panic.store(true, Ordering::Relaxed);
                            panic!(e);
                        }
                    })
                    .unwrap(),
            );
        }
        for handle in handles {
            handle.join().unwrap();
        }
    }
}

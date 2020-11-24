use crate::kernel_controller::KernelController;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::sync::mpsc::{channel, Sender};
use std::thread::{self, JoinHandle};

mod kernel_controller;
const COUNT: usize = 1024 * 1024 * 64;

fn main() -> ocl::Result<()> {
    let controller = KernelController::new()?;
    let output = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .open("primes.txt")
            .unwrap(),
    );
    let (sender, handle) = create_write_thread(output);

    let mut offset = 1i64;
    loop {
        let numbers = (offset..(COUNT as i64 * 2 + offset))
            .step_by(2)
            .collect::<Vec<i64>>();
        let primes = controller.filter_primes(numbers)?;
        sender.send(primes).unwrap();

        if (COUNT as i128 * 2 + offset as i128) > i64::MAX as i128 {
            break;
        }
        offset += COUNT as i64 * 2;
    }

    handle.join().unwrap();

    Ok(())
}

fn create_write_thread(mut writer: BufWriter<File>) -> (Sender<Vec<i64>>, JoinHandle<()>) {
    let (tx, rx) = channel();
    let handle = thread::spawn(move || {
        for primes in rx {
            for prime in primes {
                writer.write_all(format!("{}\n", prime).as_bytes()).unwrap();
            }
            writer.flush().unwrap();
        }
    });

    (tx, handle)
}

use crate::kernel_controller::KernelController;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use std::sync::mpsc::{channel, Sender};
use std::thread::{self, JoinHandle};
use std::time::Instant;
use structopt::StructOpt;

mod kernel_controller;
const COUNT: usize = 1024 * 1024 * 64;

#[derive(StructOpt, Clone, Debug)]
#[structopt()]
enum Opts {
    #[structopt(name = "calculate-primes")]
    CalculatePrimes(CalculatePrimes),
}

#[derive(StructOpt, Clone, Debug)]
struct CalculatePrimes {
    /// The number to start with
    #[structopt(default_value = "0")]
    start_offset: i64,

    /// The maximum number to calculate to
    #[structopt(default_value = "9223372036854775807")]
    max_number: i64,

    #[structopt(default_value = "primes.txt")]
    output_file: PathBuf,
}

fn main() -> ocl::Result<()> {
    let opts: Opts = Opts::from_args();
    let controller = KernelController::new()?;

    match opts {
        Opts::CalculatePrimes(prime_opts) => calculate_primes(prime_opts, controller),
    }
}

/// Calculates Prime numbers with GPU acceleration
fn calculate_primes(prime_opts: CalculatePrimes, controller: KernelController) -> ocl::Result<()> {
    let output = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .write(true)
            .open(prime_opts.output_file)
            .unwrap(),
    );
    let (sender, handle) = create_write_thread(output);

    let mut offset = prime_opts.start_offset;
    if offset % 2 == 0 {
        offset += 1;
    }
    loop {
        let start = Instant::now();
        let numbers = (offset..(COUNT as i64 * 2 + offset))
            .step_by(2)
            .collect::<Vec<i64>>();
        println!("Filtering primes from {} numbers", numbers.len());
        let primes = controller.filter_primes(numbers)?;
        println!(
            "Calculated {} primes in {} ms: {:.4} checks/s",
            primes.len(),
            start.elapsed().as_millis(),
            COUNT as f64 / start.elapsed().as_secs() as f64
        );
        sender.send(primes).unwrap();

        if (COUNT as i128 * 2 + offset as i128) > prime_opts.max_number as i128 {
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

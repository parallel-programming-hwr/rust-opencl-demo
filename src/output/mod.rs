/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */
use crate::output::csv::CSVWriter;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::mpsc::{channel, Sender};
use std::thread::{self, JoinHandle};

pub mod csv;

pub fn create_prime_write_thread(
    mut writer: BufWriter<File>,
) -> (Sender<Vec<u64>>, JoinHandle<()>) {
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

pub fn create_csv_write_thread(
    mut writer: CSVWriter<BufWriter<File>>,
) -> (Sender<Vec<String>>, JoinHandle<()>) {
    let (tx, rx) = channel();
    let handle = thread::spawn(move || {
        for row in rx {
            writer.add_row(row).unwrap();
        }
        writer.flush().unwrap();
    });

    (tx, handle)
}

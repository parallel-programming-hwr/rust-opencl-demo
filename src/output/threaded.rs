/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crossbeam_channel::Sender;
use std::io::Write;
use std::mem;
use std::thread::{self, JoinHandle};

pub struct ThreadedWriter<T>
where
    T: Send + Sync,
{
    handle: JoinHandle<()>,
    tx: Sender<T>,
}

impl<T> ThreadedWriter<T>
where
    T: Send + Sync + 'static,
{
    /// Creates a new threaded writer
    pub fn new<W, F>(mut writer: W, serializer: F) -> Self
    where
        F: Fn(T) -> Vec<u8> + Send + Sync + 'static,
        W: Write + Send + Sync + 'static,
    {
        log::trace!("Creating new threaded writer");
        let (tx, rx) = crossbeam_channel::bounded(1024);
        let handle = thread::Builder::new()
            .name("io-thread".to_string())
            .spawn(move || {
                log::trace!("Writing thread running");

                for value in rx {
                    log::trace!("Writing received value");
                    let mut bytes = serializer(value);
                    writer.write_all(&mut bytes[..]).unwrap();
                    writer.flush().unwrap();
                }
            })
            .unwrap();
        Self { handle, tx }
    }

    /// Writes a value
    pub fn write(&self, value: T) {
        log::trace!("Writing into threaded writer");
        self.tx.send(value).unwrap();
    }

    /// Closes the channel to the writer and waits for the writer thread to stop
    pub fn close(self) {
        log::trace!("Closing file and shutting down thread");
        mem::drop(self.tx);
        self.handle.join().unwrap();
    }
}

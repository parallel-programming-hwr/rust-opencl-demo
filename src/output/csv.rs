/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use crate::output::threaded::ThreadedWriter;
use std::collections::HashMap;
use std::io::Write;

pub struct ThreadedCSVWriter {
    inner: ThreadedWriter<String>,
    columns: Vec<String>,
}

impl ThreadedCSVWriter {
    /// Creates a new CSVWriter with a defined list of columns
    pub fn new<W>(writer: W, columns: &[&str]) -> Self
    where
        W: Write + Send + Sync + 'static,
    {
        let column_vec = columns
            .iter()
            .map(|column| column.to_string())
            .collect::<Vec<String>>();
        log::trace!("Creating new CSV Writer with columns: {:?}", column_vec);

        let writer = ThreadedWriter::new(writer, |v: String| v.as_bytes().to_vec());
        let mut csv_writer = Self {
            inner: writer,
            columns: column_vec.clone(),
        };
        csv_writer.add_row(column_vec);

        csv_writer
    }

    /// Adds a new row of values to the file
    pub fn add_row(&mut self, items: Vec<String>) {
        log::trace!("Adding row to CSV: {:?}", items);
        self.inner.write(
            items
                .iter()
                .fold("".to_string(), |a, b| format!("{},{}", a, b))
                .trim_start_matches(',')
                .to_string()
                + "\n",
        );
    }

    /// Adds a new row of values stored in a map to the file
    #[allow(dead_code)]
    pub fn add_row_map(&mut self, item_map: &HashMap<String, String>) {
        let mut items = Vec::new();
        for key in &self.columns {
            items.push(item_map.get(key).cloned().unwrap_or("".to_string()));
        }

        self.add_row(items)
    }

    pub fn close(self) {
        self.inner.close()
    }
}

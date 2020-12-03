/*
 * opencl demos with rust
 * Copyright (C) 2020 trivernis
 * See LICENSE for more information
 */

use std::collections::HashMap;
use std::io::{Result, Write};

pub struct CSVWriter<W: Write> {
    inner: W,
    columns: Vec<String>,
}

impl<W> CSVWriter<W>
where
    W: Write,
{
    /// Creates a new CSVWriter with a defined list of columns
    pub fn new(writer: W, columns: &[&str]) -> Self {
        Self {
            inner: writer,
            columns: columns.iter().map(|column| column.to_string()).collect(),
        }
    }

    /// Adds a new row of values to the file
    pub fn add_row(&mut self, items: Vec<String>) -> Result<()> {
        self.inner.write_all(
            items
                .iter()
                .fold("".to_string(), |a, b| format!("{},{}", a, b))
                .as_bytes(),
        )?;
        self.inner.write_all("\n".as_bytes())
    }

    /// Adds a new row of values stored in a map to the file
    #[allow(dead_code)]
    pub fn add_row_map(&mut self, item_map: &HashMap<String, String>) -> Result<()> {
        let mut items = Vec::new();
        for key in &self.columns {
            items.push(item_map.get(key).cloned().unwrap_or("".to_string()));
        }

        self.add_row(items)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.inner.flush()
    }
}

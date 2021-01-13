/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use chrono::Local;
use colored::*;
use log::{Level, LevelFilter};
use std::str::FromStr;
use std::thread;

/// Initializes the env_logger with a custom format
/// that also logs the thread names
pub fn init_logger() {
    fern::Dispatch::new()
        .format(|out, message, record| {
            let color = get_level_style(record.level());
            let mut thread_name = format!(
                "thread::{}",
                thread::current().name().unwrap_or("main").to_string()
            );
            thread_name.truncate(34);
            let mut target = record.target().to_string();
            target.truncate(39);

            out.finish(format_args!(
                "{:<20} {:<40}| {} {}: {}",
                thread_name.dimmed(),
                target.dimmed().italic(),
                Local::now().format("%Y-%m-%dT%H:%M:%S.%f"),
                record
                    .level()
                    .to_string()
                    .to_lowercase()
                    .as_str()
                    .color(color),
                message
            ))
        })
        .level(
            log::LevelFilter::from_str(
                std::env::var("RUST_LOG")
                    .unwrap_or("info".to_string())
                    .as_str(),
            )
            .unwrap_or(LevelFilter::Info),
        )
        .chain(std::io::stdout())
        .apply()
        .expect("failed to init logger");
}

fn get_level_style(level: Level) -> colored::Color {
    match level {
        Level::Trace => colored::Color::Magenta,
        Level::Debug => colored::Color::Blue,
        Level::Info => colored::Color::Green,
        Level::Warn => colored::Color::Yellow,
        Level::Error => colored::Color::Red,
    }
}

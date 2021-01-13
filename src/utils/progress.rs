/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use indicatif::{ProgressBar, ProgressStyle};
use log::LevelFilter;

pub fn get_progress_bar(size: u64) -> ProgressBar {
    if log::max_level() == LevelFilter::Info {
        let bar = ProgressBar::new(size);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("[ETA:{eta}] {bar:60.cyan/blue} {pos:>7}/{len:7} {msg}")
                .progress_chars("#>-"),
        );
        bar
    } else {
        ProgressBar::hidden()
    }
}

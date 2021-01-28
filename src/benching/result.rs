/*
 * opencl demos with rust
 * Copyright (C) 2021 trivernis
 * See LICENSE for more information
 */

use std::time::Duration;

/// Result of a benched kernel execution
#[derive(Clone, Debug)]
pub struct ProfiledResult<T>
where
    T: Send + Sync + Clone,
{
    duration: Duration,
    value: T,
}

impl<T> ProfiledResult<T>
where
    T: Send + Sync + Clone,
{
    /// Creates a new profiled result with the given duration and value
    pub fn new(duration: Duration, value: T) -> Self {
        Self { duration, value }
    }

    /// Returns the execution duration
    pub fn duration(&self) -> &Duration {
        &self.duration
    }

    /// Returns the value of the result
    pub fn value(&self) -> &T {
        &self.value
    }
}

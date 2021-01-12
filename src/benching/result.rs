use std::time::Duration;

/// Result of a benched kernel execution
#[derive(Clone, Debug)]
pub struct ProfiledResult<T>
where
    T: Send + Sync + Clone,
{
    gpu_duration: Duration,
    value: T,
}

impl<T> ProfiledResult<T>
where
    T: Send + Sync + Clone,
{
    /// Creates a new profiled result with the given duraiton and value
    pub fn new(gpu_duration: Duration, value: T) -> Self {
        Self {
            gpu_duration,
            value,
        }
    }

    /// Returns the execution duration on the gpu
    pub fn gpu_duration(&self) -> &Duration {
        &self.gpu_duration
    }

    /// Returns the value of the result
    pub fn value(&self) -> &T {
        &self.value
    }
}

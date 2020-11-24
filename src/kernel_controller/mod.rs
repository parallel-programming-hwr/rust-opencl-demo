use ocl::ProQue;

pub struct KernelController {
    pro_que: ProQue,
}

impl KernelController {
    pub fn new() -> ocl::Result<Self> {
        let pro_que = ProQue::builder()
            .src(include_str!("kernel.cl"))
            .dims(1 << 20)
            .build()?;
        Ok(Self { pro_que })
    }

    pub fn filter_primes(&self, input: Vec<i64>) -> ocl::Result<Vec<i64>> {
        lazy_static::lazy_static! {static ref PRIMES: Vec<i32> = get_lower_primes();}

        let prime_buffer = self.pro_que.buffer_builder().len(PRIMES.len()).build()?;
        prime_buffer.write(&PRIMES[..]).enq()?;

        let input_buffer = self.pro_que.buffer_builder().len(input.len()).build()?;
        input_buffer.write(&input[..]).enq()?;

        let output_buffer = self
            .pro_que
            .buffer_builder()
            .len(input.len())
            .fill_val(0u8)
            .build()?;

        let kernel = self
            .pro_que
            .kernel_builder("check_prime")
            .arg(prime_buffer.len() as i32)
            .arg(&prime_buffer)
            .arg(&input_buffer)
            .arg(&output_buffer)
            .global_work_size(input.len())
            .build()?;

        unsafe {
            kernel.enq()?;
        }

        let mut output = vec![0u8; output_buffer.len()];
        output_buffer.read(&mut output).enq()?;

        let mut input_o = vec![0i64; input_buffer.len()];
        input_buffer.read(&mut input_o).enq()?;

        Ok(input
            .iter()
            .enumerate()
            .filter(|(index, _)| output[*index] == 1)
            .map(|(_, v)| *v)
            .collect())
    }
}

/// Returns a list of prime numbers that can be used to speed up the divisibility check
fn get_lower_primes() -> Vec<i32> {
    let mut primes = Vec::new();
    let mut num = 3;

    while primes.len() < 1024 {
        let mut is_prime = true;

        if num < 3 || num % 2 == 0 {
            is_prime = false;
        } else {
            for i in (3..((num as f32).sqrt().ceil() as i32)).step_by(2) {
                if num % i == 0 {
                    is_prime = false;
                    break;
                }
            }
        }
        if is_prime {
            primes.push(num)
        }
        num += 2;
    }

    primes
}

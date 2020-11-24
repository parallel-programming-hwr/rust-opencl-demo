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

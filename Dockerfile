FROM rust as builder
RUN apt-get update
RUN apt-get install ocl-icd-opencl-dev -y
COPY . .
RUN cargo build --release

FROM nvidia/opencl:devel-ubuntu18.04
RUN apt-get update
WORKDIR benchmark
COPY --from=builder target/release/rust-opencl-demo .
ENV PATH="${PATH}:/benchmark"
ENTRYPOINT ["./rust-opencl-demo"]

FROM rust as builder
RUN apt-get update
RUN apt-get install ocl-icd-opencl-dev -y
COPY . .
RUN cargo build --release

FROM golang as builder2
RUN go get github.com/parallel-programming-hwr/http-exec

FROM nvidia/opencl:devel-ubuntu18.04
WORKDIR benchmark
COPY --from=builder2 /go/bin/http-exec .
RUN apt-get update
COPY --from=builder target/release/rust-opencl-demo .
ENV PATH="${PATH}:/benchmark"
ENTRYPOINT ["./http-exec","--command=rust-opencl-demo"]

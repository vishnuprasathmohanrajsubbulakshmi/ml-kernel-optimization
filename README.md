	# ML Kernel Optimization Project

This project implements and optimizes a fully connected (FC) neural network layer in C for embedded-style environments.

## Features

- Int8 fully connected layer implementation
- Baseline (naive) version
- Optimized version (pointer + loop improvements)
- Benchmarking framework
- Correctness tests

## Project Structure

src/
naive/ # baseline implementation
optimized/ # optimized implementations
tests/ # test and benchmark code


## How it works

The FC layer computes:

output[o] = bias[o] + sum(input[i] * weight[o][i])

- Input: int8
- Weights: int8
- Accumulator: int32
- Output: int8 (clamped)

## Results

| Version   | Time per run |
|----------|-------------|
| Naive    | ~4.7e-06 s  |
| Optimized| ~4.9e-06 s  |

(Note: On modern CPUs, compiler optimizations reduce the difference.)

## Build and Run

mkdir build
cd build
cmake ..
make
./test_fc
./bench_fc

Future Work
Loop unrolling
SIMD optimization (CMSIS-NN style)
Port to STM32 (Cortex-M)
Compare with CMSIS-NN

Author
Vishnu

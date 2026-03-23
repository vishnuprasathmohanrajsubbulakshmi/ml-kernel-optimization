#include "runtime_mlp.h"
#include "tiny_mlp_weights.h"

#define INPUT_H 28
#define INPUT_W 28
#define INPUT_SIZE (INPUT_H * INPUT_W)
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 10

static void flatten(const float *input, float *output) {
    for (int i = 0; i < INPUT_SIZE; i++) {
        output[i] = input[i];
    }
}

static void gemm(
    const float *input,
    const float *weights,
    const float *bias,
    float *output,
    int input_size,
    int output_size
) {
    for (int o = 0; o < output_size; o++) {
        float acc = bias[o];
        for (int i = 0; i < input_size; i++) {
            acc += input[i] * weights[o * input_size + i];
        }
        output[o] = acc;
    }
}

static void relu(float *x, int size) {
    for (int i = 0; i < size; i++) {
        if (x[i] < 0.0f) {
            x[i] = 0.0f;
        }
    }
}

void run_tiny_mlp(const float *input, float *output) {
    float flat[INPUT_SIZE];
    float hidden[HIDDEN_SIZE];

    flatten(input, flat);
    gemm(flat, fc1_weights, fc1_bias, hidden, INPUT_SIZE, HIDDEN_SIZE);
    relu(hidden, HIDDEN_SIZE);
    gemm(hidden, fc2_weights, fc2_bias, output, HIDDEN_SIZE, OUTPUT_SIZE);
}

#include "fc_int8.h"

static int8_t clamp_to_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

void fc_int8(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int8_t *output,
    int input_size,
    int output_size
) {
    for (int o = 0; o < output_size; o++) {
        int32_t acc = bias[o];

        for (int i = 0; i < input_size; i++) {
            acc += input[i] * weights[o * input_size + i];
        }

        output[o] = clamp_to_int8(acc);
    }
}

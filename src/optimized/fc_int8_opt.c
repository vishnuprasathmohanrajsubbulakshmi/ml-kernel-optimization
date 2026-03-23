#include "fc_int8_opt.h"

static int8_t clamp_to_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

void fc_int8_opt(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int8_t *output,
    int input_size,
    int output_size
) {
    for (int o = 0; o < output_size; o++) {
        int32_t acc = bias[o];

        const int8_t *w = &weights[o * input_size];
        const int8_t *in = input;

        for (int i = 0; i < input_size; i++) {
            acc += (*in++) * (*w++);
        }

        output[o] = clamp_to_int8(acc);
    }
}

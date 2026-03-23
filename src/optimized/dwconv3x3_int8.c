#include "dwconv3x3_int8.h"

static int8_t clamp_to_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

void dwconv2d_3x3_int8(
    const int8_t *input,
    const int8_t *kernel,
    const int32_t *bias,
    int8_t *output,
    int input_h,
    int input_w,
    int channels,
    int stride
) {
    int output_h = (input_h - 3) / stride + 1;
    int output_w = (input_w - 3) / stride + 1;

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < output_h; oh++) {
            for (int ow = 0; ow < output_w; ow++) {
                int32_t acc = bias[c];

                int ih0 = oh * stride;
                int ih1 = ih0 + 1;
                int ih2 = ih0 + 2;

                int iw0 = ow * stride;
                int iw1 = iw0 + 1;
                int iw2 = iw0 + 2;

                const int8_t *row0 = input + (ih0 * input_w * channels);
                const int8_t *row1 = input + (ih1 * input_w * channels);
                const int8_t *row2 = input + (ih2 * input_w * channels);

                acc += row0[iw0 * channels + c] * kernel[(0 * 3 + 0) * channels + c];
                acc += row0[iw1 * channels + c] * kernel[(0 * 3 + 1) * channels + c];
                acc += row0[iw2 * channels + c] * kernel[(0 * 3 + 2) * channels + c];

                acc += row1[iw0 * channels + c] * kernel[(1 * 3 + 0) * channels + c];
                acc += row1[iw1 * channels + c] * kernel[(1 * 3 + 1) * channels + c];
                acc += row1[iw2 * channels + c] * kernel[(1 * 3 + 2) * channels + c];

                acc += row2[iw0 * channels + c] * kernel[(2 * 3 + 0) * channels + c];
                acc += row2[iw1 * channels + c] * kernel[(2 * 3 + 1) * channels + c];
                acc += row2[iw2 * channels + c] * kernel[(2 * 3 + 2) * channels + c];

                int output_idx = (oh * output_w + ow) * channels + c;
                output[output_idx] = clamp_to_int8(acc);
            }
        }
    }
}

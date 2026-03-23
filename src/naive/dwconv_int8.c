#include "dwconv_int8.h"

static int8_t clamp_to_int8(int32_t x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return (int8_t)x;
}

void dwconv2d_int8(
    const int8_t *input,
    const int8_t *kernel,
    const int32_t *bias,
    int8_t *output,
    int input_h,
    int input_w,
    int channels,
    int kernel_h,
    int kernel_w,
    int stride
) {
    int output_h = (input_h - kernel_h) / stride + 1;
    int output_w = (input_w - kernel_w) / stride + 1;

    for (int c = 0; c < channels; c++) {
        for (int oh = 0; oh < output_h; oh++) {
            for (int ow = 0; ow < output_w; ow++) {
                int32_t acc = bias[c];

                for (int kh = 0; kh < kernel_h; kh++) {
                    for (int kw = 0; kw < kernel_w; kw++) {
                        int ih = oh * stride + kh;
                        int iw = ow * stride + kw;

                        int input_idx = (ih * input_w + iw) * channels + c;
                        int kernel_idx = (kh * kernel_w + kw) * channels + c;

                        acc += input[input_idx] * kernel[kernel_idx];
                    }
                }

                int output_idx = (oh * output_w + ow) * channels + c;
                output[output_idx] = clamp_to_int8(acc);
            }
        }
    }
}

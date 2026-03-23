#ifndef DWCONV_INT8_OPT_H
#define DWCONV_INT8_OPT_H

#include <stdint.h>

void dwconv2d_int8_opt(
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
);

#endif

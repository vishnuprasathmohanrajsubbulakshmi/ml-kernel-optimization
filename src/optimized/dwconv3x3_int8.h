#ifndef DWCONV3X3_INT8_H
#define DWCONV3X3_INT8_H

#include <stdint.h>

void dwconv2d_3x3_int8(
    const int8_t *input,
    const int8_t *kernel,
    const int32_t *bias,
    int8_t *output,
    int input_h,
    int input_w,
    int channels,
    int stride
);

#endif

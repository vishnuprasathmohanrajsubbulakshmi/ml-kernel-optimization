#ifndef FC_INT8_H
#define FC_INT8_H

#include <stdint.h>

void fc_int8(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int8_t *output,
    int input_size,
    int output_size
);

#endif

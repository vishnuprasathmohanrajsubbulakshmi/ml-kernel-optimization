#ifndef FC_INT8_OPT_H
#define FC_INT8_OPT_H

#include <stdint.h>

void fc_int8_opt(
    const int8_t *input,
    const int8_t *weights,
    const int32_t *bias,
    int8_t *output,
    int input_size,
    int output_size
);

#endif

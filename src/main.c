#include <stdio.h>
#include <stdint.h>
#include "naive/fc_int8.h"

int main(void) {
    int8_t input[4] = {1, 2, 3, 4};

    int8_t weights[8] = {
        1, 0, -1, 2,
        2, 1,  0, -1
    };

    int32_t bias[2] = {1, -1};
    int8_t output[2] = {0};

    fc_int8(input, weights, bias, output, 4, 2);

    printf("Output[0] = %d\n", output[0]);
    printf("Output[1] = %d\n", output[1]);

    return 0;
}

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

    int8_t expected[2] = {7, -1};

    fc_int8(input, weights, bias, output, 4, 2);

    int pass = 1;
    for (int i = 0; i < 2; i++) {
        if (output[i] != expected[i]) {
            pass = 0;
            printf("Mismatch at output[%d]: got %d, expected %d\n",
                   i, output[i], expected[i]);
        }
    }

    if (pass) {
        printf("PASS: fc_int8 output is correct.\n");
        return 0;
    } else {
        printf("FAIL: fc_int8 output is incorrect.\n");
        return 1;
    }
}

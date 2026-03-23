#include <stdio.h>
#include <stdint.h>
#include "optimized/dwconv3x3_int8.h"

int main(void) {
    const int input_h = 4, input_w = 4, channels = 2, stride = 1;
    const int output_h = 2, output_w = 2;

    int8_t input[4 * 4 * 2];
    int8_t kernel[3 * 3 * 2];
    int32_t bias[2] = {0, 0};
    int8_t output[2 * 2 * 2];

    for (int i = 0; i < 4 * 4 * 2; i++) input[i] = 1;
    for (int i = 0; i < 3 * 3 * 2; i++) kernel[i] = 1;

    dwconv2d_3x3_int8(input, kernel, bias, output, input_h, input_w, channels, stride);

    int pass = 1;
    for (int i = 0; i < output_h * output_w * channels; i++) {
        if (output[i] != 9) {
            pass = 0;
            printf("Mismatch at output[%d]: got %d, expected 9\n", i, output[i]);
        }
    }

    if (pass) {
        printf("PASS: dwconv2d_3x3_int8 output is correct.\n");
        return 0;
    } else {
        printf("FAIL: dwconv2d_3x3_int8 output is incorrect.\n");
        return 1;
    }
} 

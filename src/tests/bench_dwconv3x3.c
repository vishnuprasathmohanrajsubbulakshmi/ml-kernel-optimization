#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "optimized/dwconv3x3_int8.h"

#define INPUT_H 16
#define INPUT_W 16
#define CHANNELS 8
#define STRIDE 1
#define ITERATIONS 10000

int main(void) {
    const int output_h = (INPUT_H - 3) / STRIDE + 1;
    const int output_w = (INPUT_W - 3) / STRIDE + 1;

    int8_t input[INPUT_H * INPUT_W * CHANNELS];
    int8_t kernel[3 * 3 * CHANNELS];
    int32_t bias[CHANNELS];
    int8_t output[output_h * output_w * CHANNELS];

    for (int i = 0; i < INPUT_H * INPUT_W * CHANNELS; i++) input[i] = i % 8;
    for (int i = 0; i < 3 * 3 * CHANNELS; i++) kernel[i] = (i % 5) - 2;
    for (int i = 0; i < CHANNELS; i++) bias[i] = i;

    clock_t start = clock();

    for (int i = 0; i < ITERATIONS; i++) {
        dwconv2d_3x3_int8(input, kernel, bias, output,
                          INPUT_H, INPUT_W, CHANNELS, STRIDE);
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Total time: %f seconds\n", time_taken);
    printf("Time per run: %e seconds\n", time_taken / ITERATIONS);

    return 0;
}

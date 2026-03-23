#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "naive/dwconv_int8.h"

#define INPUT_H 16
#define INPUT_W 16
#define CHANNELS 8
#define KERNEL_H 3
#define KERNEL_W 3
#define STRIDE 1
#define ITERATIONS 10000

int main(void) {
    const int output_h = (INPUT_H - KERNEL_H) / STRIDE + 1;
    const int output_w = (INPUT_W - KERNEL_W) / STRIDE + 1;

    int8_t input[INPUT_H * INPUT_W * CHANNELS];
    int8_t kernel[KERNEL_H * KERNEL_W * CHANNELS];
    int32_t bias[CHANNELS];
    int8_t output[output_h * output_w * CHANNELS];

    for (int i = 0; i < INPUT_H * INPUT_W * CHANNELS; i++) input[i] = i % 8;
    for (int i = 0; i < KERNEL_H * KERNEL_W * CHANNELS; i++) kernel[i] = (i % 5) - 2;
    for (int i = 0; i < CHANNELS; i++) bias[i] = i;

    clock_t start = clock();

    for (int i = 0; i < ITERATIONS; i++) {
        dwconv2d_int8(input, kernel, bias, output,
                      INPUT_H, INPUT_W, CHANNELS,
                      KERNEL_H, KERNEL_W, STRIDE);
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Total time: %f seconds\n", time_taken);
    printf("Time per run: %e seconds\n", time_taken / ITERATIONS);

    return 0;
}

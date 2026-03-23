#include <stdio.h>
#include <stdint.h>
#include <time.h>
#include "optimized/fc_int8_opt.h"

#define INPUT_SIZE 64
#define OUTPUT_SIZE 32
#define ITERATIONS 100000

int main(void) {
    int8_t input[INPUT_SIZE];
    int8_t weights[INPUT_SIZE * OUTPUT_SIZE];
    int32_t bias[OUTPUT_SIZE];
    int8_t output[OUTPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; i++) input[i] = i % 8;
    for (int i = 0; i < INPUT_SIZE * OUTPUT_SIZE; i++) weights[i] = (i % 5) - 2;
    for (int i = 0; i < OUTPUT_SIZE; i++) bias[i] = i;

    clock_t start = clock();

    for (int i = 0; i < ITERATIONS; i++) {
        fc_int8_opt(input, weights, bias, output, INPUT_SIZE, OUTPUT_SIZE);
    }

    clock_t end = clock();

    double time_taken = (double)(end - start) / CLOCKS_PER_SEC;

    printf("Total time: %f seconds\n", time_taken);
    printf("Time per run: %e seconds\n", time_taken / ITERATIONS);

    return 0;
}

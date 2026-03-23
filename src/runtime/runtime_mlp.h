#ifndef RUNTIME_MLP_H
#define RUNTIME_MLP_H

void run_tiny_mlp(
    const float *input,   // shape: [1, 1, 28, 28]
    float *output         // shape: [1, 10]
);

#endif

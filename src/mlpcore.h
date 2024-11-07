#ifndef MLPCORE_H_
#define MLPCORE_H_

#include <stdlib.h>

typedef enum {
    SIGMOID,
    STEP,
} activation_t;

typedef struct {
    size_t n_inputs, n_outputs, n_layers;
    size_t n_weights;
    union {
        size_t n_biases, n_neurons;
    };
    size_t *layers_shape;
    double *weights, *biases, *sum, *results;
    activation_t *activation;
} mlp_t;

int mlp_init(mlp_t *const mlp, const size_t *const layers_shape,
               const size_t n_layers, const activation_t activation);
void mlp_deinit(mlp_t *const mlp);
void mlp_seed(void);
int mlp_weights_init(mlp_t *const mlp, const double weights_range[2]);
void mlp_print_weights(const mlp_t *const mlp);
int mlp_feedforward(mlp_t *const mlp, const double *const input,
                    double *const output);
int mlp_backpropagation(mlp_t *const mlp, const double learning_rate,
                        const double *const input, const size_t n_samples,
                        const double *const output_expected, 
                        double *const error_ptr);

#endif

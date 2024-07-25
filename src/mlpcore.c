#include "mlpcore.h"
#include <stdio.h>
#include <errno.h>
#include <string.h>


int mlp_config(mlp_t *const mlp, const  size_t *const layers_shape, 
               const size_t n_layers, const activation_t activation)
{
    int status = 0;

    if (mlp == NULL || layers_shape == NULL) {
        fprintf(stderr, "ERROR: null pointers passes\n");
        status = -EINVAL;
        goto __exit;
    }

    if (n_layers < 2) {
        fprintf(stderr, "ERROR: n_layers must be at least 2\n");
        status = -EINVAL;
        goto __exit;
    }

    size_t n_weights = 0;
    size_t n_biases = 0;

    for (size_t i = 1; i < n_layers; i++) {
        n_weights += layers_shape[i] * layers_shape[i-1];
        n_biases += layers_shape[i];
    }

    mlp->layers_shape = malloc(n_layers * sizeof(*mlp->layers_shape));
    mlp->weights = malloc(n_weights * sizeof(*mlp->weights));
    mlp->biases = malloc(n_biases * sizeof(*mlp->biases));
    mlp->results = malloc(n_biases * sizeof(*mlp->results));
    mlp->sum = malloc(n_biases * sizeof(*mlp->sum));
    mlp->activation = malloc(n_biases * sizeof(*mlp->activation));

    if (mlp->layers_shape == NULL || mlp->weights == NULL ||
        mlp->biases == NULL || mlp->results == NULL || 
        mlp->sum == NULL || mlp->activation == NULL) {
        fprintf(stderr, "ERROR: it was not possible to allocate memory\n");
        status = -ENOMEM;
        mlp_deconf(mlp);
        goto __exit;
    }

    mlp->n_weights = n_weights;
    mlp->n_biases = n_biases;
    mlp->n_layers = n_layers;
    mlp->n_inputs = layers_shape[0];
    mlp->n_outputs = layers_shape[n_layers - 1];

    memcpy(mlp->layers_shape, layers_shape, n_layers * sizeof(*layers_shape));
    memset(mlp->activation, activation, n_biases * sizeof(*mlp->activation));

__exit:
    return status;
}


void mlp_deconf(mlp_t *mlp)
{
    free(mlp->layers_shape);
    free(mlp->weights);
    free(mlp->biases);
    free(mlp->results);
    free(mlp->sum);
    free(mlp->activation);

    memset(mlp, 0, sizeof(mlp_t));
}


int mlp_weights_init(double *array, size_t n, const double weights_range[2])
{
    int status = 0;
    double max_val, min_val;

    if (array == NULL || weights_range == NULL) {
        fprintf(stderr, "ERROR: null pointer passed\n");
        status = -EINVAL;
        goto __exit;
    }

    if (weights_range[0] > weights_range[1]) {
        max_val = weights_range[0];
        min_val = weights_range[1];
    }
    else {
        max_val = weights_range[1];
        min_val = weights_range[0];
    }

    for (size_t i = 0; i < n; i++) {
        int rand_val = rand();
        array[i] = min_val + ((double) rand_val * (max_val - min_val) / 
                              (double) (RAND_MAX));
    }



__exit:
    return status;
}


static inline double _dot_product(double *v1, double *v2, size_t n)
{
    double result = 0.;

    for (size_t i = 0; i < n; i++) {
        result += v1[i] * v2[i];
    }

    return result;
}


static inline double activate(double x, activation_t activation)
{
    double result = 0;
    switch (activation) {
        case STEP:
        result = (x > 0.) ? 1. : 0.;
    }

    return result;
}

int mlp_feedforward(mlp_t *mlp, double *input, double *output)
{
    int status = 0;
    double *x = input;
    double *weights;
    size_t x_displace = 0;
    size_t neuron_global = 0;

    if (mlp == NULL || input == NULL || output == NULL) {
        fprintf(stderr, "ERROR: null pointer passed\n");
        status = -EINVAL;
        goto __exit;
    }

    weights = mlp->weights;
    for (size_t layer = 0; layer < mlp->n_layers - 1; layer++) {
        for (size_t neuron = 0; neuron < mlp->layers_shape[layer+1]; neuron++) {
            double temp_sum, temp_result;
            temp_sum = _dot_product(x, weights, mlp->layers_shape[layer]) +
                       mlp->biases[neuron_global];
            mlp->sum[neuron_global] = temp_sum;
            
            temp_result = activate(temp_sum, mlp->activation[neuron_global]);
            mlp->results[neuron_global] = temp_result;

            weights += mlp->layers_shape[layer];
            neuron_global++;
        }
        x = mlp->results + x_displace;
        x_displace += mlp->layers_shape[layer + 1]; 
    }

    size_t n_neurons = mlp->n_neurons;
    size_t n_outputs = mlp->n_outputs;
    memcpy(output, &mlp->results[n_neurons - n_outputs], 
           n_outputs * sizeof(*mlp->results));

__exit:

    return status;
}

#include "mlpcore.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>


void mlp_seed(void)
{
    srand(time(0));
}


int mlp_init(mlp_t *const mlp, const size_t *const layers_shape,
               const size_t n_layers, const activation_t activation) {
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
        n_weights += layers_shape[i] * layers_shape[i - 1];
        n_biases += layers_shape[i];
    }

    mlp->layers_shape = malloc(n_layers * sizeof(*mlp->layers_shape));
    mlp->weights = malloc(n_weights * sizeof(*mlp->weights));
    mlp->biases = malloc(n_biases * sizeof(*mlp->biases));
    mlp->results = malloc(n_biases * sizeof(*mlp->results));
    mlp->sum = malloc(n_biases * sizeof(*mlp->sum));
    mlp->activation = malloc(n_biases * sizeof(*mlp->activation));

    if (mlp->layers_shape == NULL || mlp->weights == NULL ||
        mlp->biases == NULL || mlp->results == NULL || mlp->sum == NULL ||
        mlp->activation == NULL) {
        fprintf(stderr, "ERROR: it was not possible to allocate memory\n");
        status = -ENOMEM;
        mlp_deinit(mlp);
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

void mlp_deinit(mlp_t *const mlp) {
    free(mlp->layers_shape);
    free(mlp->weights);
    free(mlp->biases);
    free(mlp->results);
    free(mlp->sum);
    free(mlp->activation);

    memset(mlp, 0, sizeof(mlp_t));
}

int __mlp_weights_init(double *const array, const size_t n,
                     const double weights_range[static 2]) {
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
    } else {
        max_val = weights_range[1];
        min_val = weights_range[0];
    }

    for (size_t i = 0; i < n; i++) {
        int rand_val = rand();
        array[i] = min_val + ((double)rand_val * (max_val - min_val) /
                              (double)(RAND_MAX));
    }

__exit:
    return status;
}


int mlp_weights_init(mlp_t *const mlp, const double weights_range[2])
{
    int status = 0;
    status |= __mlp_weights_init(mlp->weights, mlp->n_weights, weights_range);
    status |= __mlp_weights_init(mlp->biases, mlp->n_biases, weights_range);

    return status;
}


void mlp_print_weights(const mlp_t *const mlp) 
{
    if (mlp == NULL) {
        fprintf(stderr, "ERROR: Null pointer passed\n");
        return;
    }

    for (size_t i = 0; i < mlp->n_weights; i++) {
        printf("w[%lu] = %f,   ", i, mlp->weights[i]);
    }
    printf("\n");
    for (size_t i = 0; i < mlp->n_biases; i++) {
        printf("b[%lu] = %f,   ", i, mlp->biases[i]);
    }
    printf("\n");
}


static inline double _dot_product(const double *const v1,
                                  const double *const v2, const size_t n) {
    double result = 0.;

    for (size_t i = 0; i < n; i++) {
        result += v1[i] * v2[i];
    }

    return result;
}

static double activate(double x, activation_t activation)
{
    double result;

    switch (activation) {
    case SIGMOID:
        result = 1. / (1. + exp(-x));
        break;
    case STEP:
        result = (x > 0.) ? 1. : 0.;
        break;
    default:
        result = 0.;
    }

    return result;
}

int mlp_feedforward(mlp_t *const mlp, const double *const input,
                    double *const output) {
    int status = 0;
    const double *x = input;
    double *weights;
    size_t x_displace = 0;
    size_t neuron_global = 0;

    if (mlp == NULL || input == NULL || output == NULL) {
        fprintf(stderr, "ERROR: null pointer passed\n");
        status = -EINVAL;
        goto __exit;
    }

    weights = mlp->weights;
    for (size_t layer_i = 0; layer_i < mlp->n_layers - 1; layer_i++) {
        for (size_t neuron = 0; neuron < mlp->layers_shape[layer_i + 1];
             neuron++) {
            double temp_sum, temp_result;
            temp_sum = _dot_product(x, weights, mlp->layers_shape[layer_i]) +
                       mlp->biases[neuron_global];
            mlp->sum[neuron_global] = temp_sum;

            temp_result = activate(temp_sum, mlp->activation[neuron_global]);
            mlp->results[neuron_global] = temp_result;

            weights += mlp->layers_shape[layer_i];
            neuron_global++;
        }
        x = mlp->results + x_displace;
        x_displace += mlp->layers_shape[layer_i + 1];
    }

    size_t n_neurons = mlp->n_neurons;
    size_t n_outputs = mlp->n_outputs;
    memcpy(output, &mlp->results[n_neurons - n_outputs],
           n_outputs * sizeof(*mlp->results));

__exit:
    return status;
}


static double der_activate(const double x, const activation_t activation)
{
    double result;

    double aux;
    switch (activation) {
    case SIGMOID:
        aux = 1. / (1. + exp(-x));
        result = aux * (1. - aux);
        break;
    case STEP:
        result = (fabs(x) < 1e-12) ? 1. : 0.;
        break;
    default:
        result = 0.;
    }

    return result;
}


double mlp_cost_function(const mlp_t *const mlp, const double *const output,
                         const double *const output_expected)
{
    double sum = 0.;
    for (size_t i = 0; i < mlp->n_outputs; i++) {
        double error;
        error = output_expected[i] - output[i];
        sum += error * error;
    }
    return sum;
}


static void _mlp_update_params(double *const params, const size_t array_size,
                               const double learning_rate,
                               const size_t n_samples,
                               const double *const der_params)
{
    const double aux = learning_rate / (double) n_samples;
    for (size_t i = 0; i < array_size; i++) {
        params[i] -= aux * der_params[i];
    }
}


static void _mlp_update(mlp_t *const mlp, const double learning_rate, 
                        const size_t n_samples, 
                        const double *const der_weights,
                        const double *const der_biases)
{

    _mlp_update_params(mlp->weights, mlp->n_weights, learning_rate, 
                       n_samples, der_weights);
    _mlp_update_params(mlp->biases, mlp->n_biases, learning_rate, 
                       n_samples, der_biases);
}


int mlp_backpropagation(mlp_t *const mlp, const double learning_rate,
                        const double *const input, const size_t n_samples,
                        const double *const output_expected, 
                        double *const error_ptr)
{
    int status = 0;
    double error_total = 0.;

    if (mlp == NULL || input == NULL || output_expected == NULL || 
        n_samples == 0) {
        fprintf(stderr, "ERROR: Invalid input arguments\n");
        status = -EINVAL;
        goto __exit;
    }

    /* Create buffers for derivatives */
    double *const der_weights = calloc(mlp->n_weights, sizeof(*mlp->weights));
    double *const der_biases = calloc(mlp->n_biases, sizeof(*mlp->biases));
    double *const errors = calloc(mlp->n_neurons, sizeof(*mlp->results));
    double *const output = malloc(mlp->n_outputs * sizeof(*mlp->results));

    if (der_weights == NULL || der_biases == NULL || errors == NULL || 
        output == NULL) {
        fprintf(stderr, "ERROR: It was not possible to allocate memory\n");
        status = -ENOMEM;
        goto __dealloc;
    }

    /* Auxiliar constants */
    const size_t n_layers = mlp->n_layers;
    const size_t n_weights = mlp->n_weights;
    const size_t n_neurons = mlp->n_neurons;
    const size_t n_inputs = mlp->n_inputs;
    const size_t n_outputs = mlp->n_outputs;
    const size_t *const layers_shape = mlp->layers_shape;

    /* Begin training by looping through samples */
    for (size_t sample = 0; sample < n_samples; sample++) {
        mlp_feedforward(mlp, &input[n_inputs * sample], output);

        if (error_ptr != NULL) {
            error_total += mlp_cost_function(mlp, output, 
                                        &output_expected[sample * n_outputs]);
        }

        /* Auxiliar variables */
        size_t layer_size = n_outputs;
        size_t neuron_i = n_neurons - layer_size;
        size_t weight_i = n_weights - 
            layers_shape[n_layers - 1] * layers_shape[n_layers - 2];
        size_t layer_size_prev, layer_size_next;

        /* Loop through neurons of last layer */
        for (size_t i = 0; i <  layer_size; i++) {
            double y_expected, delta, sum_aux;
            activation_t activ_aux;

            y_expected = output_expected[i + sample * n_outputs];
            errors[neuron_i + i] = output[i] - y_expected;

            sum_aux = mlp->sum[neuron_i + i];
            activ_aux = mlp->activation[neuron_i + i];

            delta = errors[neuron_i + i] * der_activate(sum_aux, activ_aux);
            der_biases[neuron_i + i] += delta;

            /* Loop through neurons' inputs */
            layer_size_prev = layers_shape[n_layers - 2];
            for (size_t j = 0; j < layer_size_prev; j++) {
                size_t aux1, aux2;
                double result;
                aux1 = j + neuron_i - layer_size_prev;
                aux2 = j + weight_i + i *layer_size_prev;

                result = mlp->results[aux1];
                der_weights[aux2] += delta * result;
            }
        }

        /* Backpropagate erros of last layer to hidden ones */
        for (size_t layer_i = n_layers - 2; layer_i > 0; layer_i--) {
            layer_size = layers_shape[layer_i];
            layer_size_prev = layers_shape[layer_i - 1];
            layer_size_next = layers_shape[layer_i + 1];

            weight_i -= layer_size * layer_size_prev;
            neuron_i -= layer_size;

            /* Loop through neurons of layer */
            for (size_t i = 0; i< layer_size; i++) {
                double sum_aux = 0.;
                for (size_t j = 0; j < layer_size_next; j++) {
                    size_t aux1, aux2;
                    /* Index of first weight of next layer */
                    aux1 = i + layer_size * (j + layer_size_prev) + weight_i;
                    /* Index of neurons of next layer */
                    aux2 = j + neuron_i + layer_size;

                    sum_aux += mlp->weights[aux1] * errors[aux2];
                }

                activation_t activ_aux;
                double delta, sum_aux2;

                activ_aux = mlp->activation[neuron_i + i];
                sum_aux2 = mlp->sum[neuron_i + i];

                delta = sum_aux * der_activate(sum_aux2, activ_aux);

                errors[neuron_i + i] = delta;
                der_biases[neuron_i + i] += delta;

                const double *x;
                if (layer_i > 1) x = &mlp->results[neuron_i - layer_size_prev];
                else x = &input[sample * n_inputs];

                for (size_t j = 0; j < layer_size_prev; j++) {
                    size_t aux;
                    aux = j + i * layer_size_prev + weight_i;
                    der_weights[aux] += delta * x[j];
                }
            }
        }
    }

    _mlp_update(mlp, learning_rate, n_samples, der_weights, der_biases);

    if (error_ptr != NULL) *error_ptr = error_total;


__dealloc:
    free(output);
    free(errors);
    free(der_biases);
    free(der_weights);
__exit:
    return status;
}

#include "../src/mlpcore.h"
#include <stdio.h>


#define ARRAY_LEN(array) (sizeof(array) / sizeof(array[0]))

int main(void)
{
    mlp_t mlp;
    int status = 0;
    const size_t shape[] = {2, 2, 1};
    const size_t n_layers = ARRAY_LEN(shape);
    const double range[] = {-1, 1};

    const double x[][2] = {{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}};
    const double y[] = {0., 1., 1., 0};

    status |= mlp_config(&mlp, shape, n_layers, STEP);

//    status |= mlp_weights_init(mlp.weights, mlp.n_weights, range);
//    status |= mlp_weights_init(mlp.biases, mlp.n_biases, range);
    //
    mlp.weights[0] = 1.;
    mlp.weights[1] = 1.;
    mlp.weights[2] = 1.;
    mlp.weights[3] = 1.;
    mlp.weights[4] = -1.;
    mlp.weights[5] = 1.;

    mlp.biases[0] = -1.5;
    mlp.biases[1] = -0.5;
    mlp.biases[2] = -0.5;

    double output;

    for (size_t sample = 0; sample < 4; sample++) {
        mlp_feedforward(&mlp, x[sample], &output);

        printf("a = %d, b = %d, out = %f, expected = %d\n",
               (int) x[sample][0], (int) x[sample][1], output, (int) y[sample]);
    }

    mlp_deconf(&mlp);

    return status;
}

#include "../src/mlpcore.h"
#include <stdio.h>
#include <time.h>


#define ARRAY_LEN(array) (sizeof(array) / sizeof(array[0]))

int test_xor(void);


int main(void) {
    int status = 0;

    srand(time(0));
    
    status |= test_xor();
    
    return status;
}

int test_xor(void) {
    int status = 0;
    printf("********** XOR **********\n");
    const double x[][2] = {{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}};
     const double y[] = {0., 1., 1., 0};
  
    mlp_t mlp;
    const size_t shape[] = {2, 2, 1};
    const size_t n_layers = ARRAY_LEN(shape);
    const double weights_range[2] = {-1., 1};
  
    status |= mlp_init(&mlp, shape, n_layers, SIGMOID);
    status |= mlp_weights_init(&mlp, weights_range);

    const size_t EPOCH_MAX = 100000;
    const double learning_rate = .2;

    printf("********* TRAINING ********\n");
    double error;
    for (size_t epoch = 1; epoch <= EPOCH_MAX; epoch++) {
        mlp_backpropagation(&mlp, learning_rate, &x[0][0], 4, y, &error);
        if (epoch % (EPOCH_MAX / 5) == 0) {
            printf("Epoch: %lu, error = %.10lf\n", epoch, error);
        }
    }

    printf("********* TESTING *********\n");
    double output, error_total = 0.;
    for (size_t sample = 0; sample < 4; sample++) {
        mlp_feedforward(&mlp, x[sample], &output);
        error = output - y[sample];
        error_total += error * error;
        printf("a = %d, b = %d, out = %f, expected = %d, error = %f\n",
               (int) x[sample][0], (int) x[sample][1], output, (int) y[sample],
               error);
    }
    printf("\nTotal error = %.10lf\n", error_total);

    mlp_print_weights(&mlp);
  
    mlp_deinit(&mlp);
    printf("*************************\n");
  
    return status;
}


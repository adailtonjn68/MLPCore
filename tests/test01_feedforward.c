#include "../src/mlpcore.h"
#include <stdio.h>


#define ARRAY_LEN(array) (sizeof(array) / sizeof(array[0]))

int test_xor(void);
int test_and(void);
int test_or(void);
int test_and_or(void);

int main(void)
{
    int status = 0;

    status |= test_xor();
    status |= test_and();
    status |= test_or();
    status |= test_and_or();

    return status;
}


int test_xor(void)
{
    int status = 0;
    printf("********** XOR **********\n");
    const double x[][2] = {{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}};
    const double y[] = {0., 1., 1., 0};

    mlp_t mlp;
    const size_t shape[] = {2, 2, 1};
    const size_t n_layers = ARRAY_LEN(shape);

    status |= mlp_init(&mlp, shape, n_layers, STEP);

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

    mlp_deinit(&mlp);
    printf("*************************\n");

    return status;
}


int test_and(void)
{
    int status = 0;
    printf("********** AND **********\n");
    const double x[][2] = {{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}};
    const double y[] = {0., 0., 0., 1};

    mlp_t mlp;
    const size_t shape[] = {2, 1};
    const size_t n_layers = ARRAY_LEN(shape);

    status |= mlp_init(&mlp, shape, n_layers, STEP);

    mlp.weights[0] = 1.;
    mlp.weights[1] = 1.;

    mlp.biases[0] = -1.5;

    double output;

    for (size_t sample = 0; sample < 4; sample++) {
        mlp_feedforward(&mlp, x[sample], &output);

        printf("a = %d, b = %d, out = %f, expected = %d\n",
               (int) x[sample][0], (int) x[sample][1], output, (int) y[sample]);
    }

    mlp_deinit(&mlp);
    printf("*************************\n");

    return status;
}


int test_or(void)
{
    int status = 0;
    printf("********** OR ***********\n");
    const double x[][2] = {{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}};
    const double y[] = {0., 1., 1., 1};

    mlp_t mlp;
    const size_t shape[] = {2, 1};
    const size_t n_layers = ARRAY_LEN(shape);

    status |= mlp_init(&mlp, shape, n_layers, STEP);

    mlp.weights[0] = 1.;
    mlp.weights[1] = 1.;

    mlp.biases[0] = 0.;

    double output;

    for (size_t sample = 0; sample < 4; sample++) {
        mlp_feedforward(&mlp, x[sample], &output);

        printf("a = %d, b = %d, out = %f, expected = %d\n",
               (int) x[sample][0], (int) x[sample][1], output, (int) y[sample]);
    }

    mlp_deinit(&mlp);
    printf("*************************\n");

    return status;
}


int test_and_or(void)
{
    int status = 0;
    printf("******* AND-OR **********\n");
    const double x[][2] = {{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}};
    const double y[][2] = {{0, 0},
                           {1, 0},
                           {1, 0},
                           {1, 1}};

    mlp_t mlp;
    const size_t shape[] = {2, 2};
    const size_t n_layers = ARRAY_LEN(shape);

    status |= mlp_init(&mlp, shape, n_layers, STEP);

    mlp.weights[0] = 1.;
    mlp.weights[1] = 1.;
    mlp.weights[2] = 1.;
    mlp.weights[3] = 1.;

    mlp.biases[0] = 0.;
    mlp.biases[1] = -1.5;

    double output[2];

    for (size_t sample = 0; sample < 4; sample++) {
        mlp_feedforward(&mlp, x[sample], output);

        printf("a = %d, b = %d, out = {%.3f, %.3f}, expected = {%d, %d}\n",
               (int) x[sample][0], (int) x[sample][1], output[0], output[1], (int) y[sample][0], (int) y[sample][1]);
    }

    mlp_deinit(&mlp);
    printf("*************************\n");

    return status;
}

#include "../src/mlpcore.h"
#include <stdio.h>


#define ARRAY_LEN(array) (sizeof(array) / sizeof(array[0]))

int test_xor(void);


int main(void) {
    int status = 0;
    
    status |= test_xor();
    
    return status;
}

int test_xor(void) {
    int status = 0;
    printf("********** XOR **********\n");
    //const double x[][2] = {{0., 0.}, {0., 1.}, {1., 0.}, {1., 1.}};
    // const double y[] = {0., 1., 1., 0};
  
    mlp_t mlp;
    const size_t shape[] = {2, 2, 1};
    const size_t n_layers = ARRAY_LEN(shape);
    const double weights_range[2] = {-1., 1};
  
    status |= mlp_init(&mlp, shape, n_layers, STEP);
    mlp_seed();
    status |= mlp_weights_init(&mlp, weights_range);

    mlp_print_weights(&mlp);
  
    mlp_deinit(&mlp);
    printf("*************************\n");
  
    return status;
}


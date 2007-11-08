#include "kernel_t.h"

__global__ void global_function (int i, int *j) {
    *j = 666;
}

const char* get_kernel() {
    return ((const char*) global_function);
}




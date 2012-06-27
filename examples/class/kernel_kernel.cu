/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#include "kernel_t.h"

__global__ void global_function (test_device a) {
    
}

kernelT get_kernel() {
	return global_function;
}

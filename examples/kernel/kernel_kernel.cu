/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#include "kernel_t.h"

__global__ void global_function (const int i, int &j) {
	j = i;
}

kernelT get_kernel() {
	return (kernelT)global_function;
}

/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#include "cupp/deviceT/vector.h"
#include "kernel_t.h"

__global__ void global_function (cupp::deviceT::vector<int> &i) {
	i[threadIdx.x] *= 2;
}

kernelT get_kernel() {
	return (kernelT)global_function;
}

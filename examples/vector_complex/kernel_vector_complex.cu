/*
 * Author:  Jens Breitbart, http://www.gpuified.de/contact/
 *
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#include "deviceT/vector.h"
#include "kernel_t.h"

using namespace cupp;

__global__ void global_function (deviceT::vector< deviceT::vector <int> > &i) {
	i[threadIdx.x][0] *= 2;
}

kernelT get_kernel() {
	return global_function;
}

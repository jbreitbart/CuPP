/*
 * Author:  Jens Breitbart, http://www.gpuified.de/contact/
 *
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#include "cupp/deviceT/memory1d.h"

#include "kernel_t.h"

__global__ void global_function (cupp::deviceT::memory1d<int>& p) {
	p[threadIdx.x]*=2;
}

kernelT get_kernel() {
	return global_function;
}

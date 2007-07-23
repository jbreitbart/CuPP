/*
 * Author:  Jens Breitbart, http://www.gpuified.de/contact/
 *
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#include "deviceT/memory1d.h"

__global__ void global_function (cupp::deviceT::memory1d<int> p) {
	p[threadIdx.x]*=2;
}

typedef void(*kernelT)(cupp::deviceT::memory1d<int>);

kernelT get_kernel() {
	return global_function;
}

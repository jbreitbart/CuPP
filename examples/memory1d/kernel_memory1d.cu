/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#include "cupp/deviceT/memory1d.h"

#include "kernel_t.h"

__global__ void global_function (cupp::deviceT::memory1d<int>* p_) {
	__shared__ cupp::deviceT::memory1d<int> p;
	if (threadIdx.x==0) p = *p_;
	__syncthreads();
	
	p[threadIdx.x]*=2;
}

kernelT get_kernel() {
	return global_function;
}

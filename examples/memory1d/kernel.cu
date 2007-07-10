/*
 * Author:  Jens Breitbart, http://www.gpuified.de/contact/
 *
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#include "deviceT/memory1d.h"

__global__ void global_function (cupp::deviceT::memory1d<int> p) {
	#if defined (WE_WANT_OPENMP)
		Loop over all grid cells
			Loop over all blocks in parallel

			To get it working we would just need to:
			- rewrite the kernel operator()
			- replace __synchthreads() with a omp barrier
			- feed this function with the grid/block size
			- and cope with millions of differentes between CUDA and the OpenMP Standard (or let the developer deal with it)
	#endif
	p[threadIdx.x]*=2;
}

typedef void(*kernelT)(cupp::deviceT::memory1d<int>);

kernelT get_kernel() {
	return global_function;
}

/*
 * Author:  Jens Breitbart, http://www.gpuified.de/contact/
 *
 * Copyright: See COPYING file that comes with this distribution
 *
 */
// includes, system
#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "memory1d.h"

__global__ void real_GPU (int* p) {
	p[threadIdx.x]*=2;
}

void kernel(cupp::memory1d<int> &p) {
	using namespace cupp;
	// set up the enviroment
	dim3 block_dim (8);
	dim3 grid_dim  (1);
	
	// start the kernel
	real_GPU <<< grid_dim, block_dim >>> (p.cuda_pointer());

	cudaError_t err = cudaGetLastError();
	if( cudaSuccess != err) {
		fprintf( stderr, "Cuda error: KERNEL EXEC FAILED in file '%s' in line %i : %s.\n",
		__FILE__, __LINE__, cudaGetErrorString( err) );
		exit(EXIT_FAILURE);
	}

}

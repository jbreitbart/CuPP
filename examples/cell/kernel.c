/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#include <ea.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>

#include <stdio.h>

#include "cupp/cell/cuda_stub.h"

static struct dim3 gridDim;
static struct dim3 blockDim;
static struct dim3 blockIdx;

typedef int local_int[512];


#define LOCAL(a) a[threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y]
#define __shared__

#define __syncthreads()	\
	} } } \
	for (threadIdx.x=0; threadIdx.x<blockDim.x; ++threadIdx.x) { \
	for (threadIdx.y=0; threadIdx.y<blockDim.y; ++threadIdx.y) { \
	for (threadIdx.z=0; threadIdx.z<blockDim.z; ++threadIdx.z) {

#define START \
	for (threadIdx.x=0; threadIdx.x<blockDim.x; ++threadIdx.x) { \
	for (threadIdx.y=0; threadIdx.y<blockDim.y; ++threadIdx.y) { \
	for (threadIdx.z=0; threadIdx.z<blockDim.z; ++threadIdx.z) {

#define END } } }

static inline void kernel (__ea int* arr) {
	struct dim3 threadIdx;

	__shared__ int shared[blockIdx.x];
	local_int temp;

	START

	LOCAL(temp) = 2;

	arr[threadIdx.x + blockIdx.x*blockDim.x] *= LOCAL(temp);

	END


}

typedef union {
	__ea int* arg;
	char dummy[sizeof(__ea int*)];
} arg1T;

int main () {
	int i;
	__ea char* stack_ptr = (__ea char*) spu_read_in_mbox();
	const unsigned int start_calc = spu_read_in_mbox();
	const unsigned int end_calc = spu_read_in_mbox();


	char stack[2*sizeof(struct dim3) + 256];

	for (i=0; i<2*sizeof(struct dim3) + 256; ++i) {
		stack[i] = stack_ptr[i];
	}

	// 1. get grid- and blockdim
	gridDim  = *(struct dim3*) stack;
	blockDim = *(struct dim3*) (stack + sizeof(struct dim3));

	// 2. get the arguments out of the stack
	arg1T arg1;
	for (i=0; i<sizeof(__ea int*); ++i) {
		arg1.dummy[i] = stack[2*sizeof(struct dim3) + i];
	}



	for (blockIdx.x=start_calc; blockIdx.x < end_calc; ++blockIdx.x) {
	for (blockIdx.y=0; blockIdx.y<gridDim.y; ++blockIdx.y) {
	for (blockIdx.z=0; blockIdx.z<gridDim.z; ++blockIdx.z) {

		kernel (arg1.arg);

	}
	}
	}

	__cache_flush();
	spu_write_out_mbox(0);

	return 0;
}

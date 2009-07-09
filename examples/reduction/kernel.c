/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#include <ea.h>
#include <spu_intrinsics.h>
#include <spu_mfcio.h>
#include <libsync.h>

#include <stdio.h>

#include "cupp/cell/cuda_stub.h"

#include "kernel_t.h"

static struct dim3 gridDim;
static struct dim3 blockDim;
static struct dim3 blockIdx;

static int last_block = 0;
static datatype atomic_result = 0;

typedef datatype local_datatype[512];


#define LOCAL(a) #a[threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y]
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

static inline void kernel (__ea datatype* arr, __ea datatype* result) {
	struct dim3 threadIdx;

	START

	atomic_result+= arr[threadIdx.x + blockIdx.x*blockDim.x];

	END

	if (last_block==1) {
		atomic_add(atomic_result, (unsigned int)result);
	}
}


typedef union {
	__ea int* arg;
	char dummy[sizeof(__ea int*)];
} arg1T;

typedef union {
	__ea int* arg;
	char dummy[sizeof( int*)];
} arg2T;


int main () {
while (1==1) {

	int stack_used = 0;
	__ea char* stack_ptr = (__ea char*) spu_read_in_mbox();
	const unsigned int start_calc = spu_read_in_mbox();
	const unsigned int end_calc = spu_read_in_mbox();

	int i;

	char stack[2*sizeof(struct dim3) + 256];

	for (i=0; i<2*sizeof(struct dim3) + 256; ++i) {
		stack[i] = stack_ptr[i];
	}

	// 1. get grid- and blockdim
	gridDim  = *(struct dim3*) stack;
	blockDim = *(struct dim3*) (stack + sizeof(struct dim3));

	stack_used = 2*sizeof(struct dim3);

	// 2. get the arguments out of the stack
	arg1T arg1;
	for (i=0; i<sizeof(__ea int*); ++i) {
		arg1.dummy[i] = stack[stack_used + i];
	}
	stack_used += sizeof(__ea int*);

	arg2T arg2;
	for (i=0; i<sizeof(__ea int*); ++i) {
		arg2.dummy[i] = stack[stack_used + i];
	}
	stack_used += sizeof(__ea int*);

	last_block = 0;
	atomic_result = 0;

	for (blockIdx.x=start_calc; blockIdx.x < end_calc; ++blockIdx.x) {
	for (blockIdx.y=0; blockIdx.y<gridDim.y; ++blockIdx.y) {
	for (blockIdx.z=0; blockIdx.z<gridDim.z; ++blockIdx.z) {

		if ( blockIdx.x==end_calc-1 && blockIdx.y==gridDim.y-1 && blockIdx.z==gridDim.z-1)
			last_block = 1;

		kernel (arg1.arg, arg2.arg);

	}
	}
	}


	__cache_flush();
	spu_write_out_mbox(0);
}

	return 0;
}

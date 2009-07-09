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

static inline void kernel (__ea datatype* A, __ea datatype* B, __ea datatype* C, const int size) {
	struct dim3 threadIdx;

	START

	int thread_row = (blockDim.y * blockIdx.y) + threadIdx.y;
	int thread_col = (blockDim.x * blockIdx.x) + threadIdx.x;
	
	int c_elem = 0;
	int i=0;
	
	for (i = 0; i < size; i++) {
		c_elem += A[(size * thread_row) + i] * B[(size * i) + thread_col];
	}
	
	C[(size * thread_row) + thread_col] = c_elem;


	END
}


typedef union {
	__ea datatype* arg;
	char dummy[sizeof(__ea datatype*)];
} arg1T;

typedef union {
	__ea datatype* arg;
	char dummy[sizeof(__ea datatype*)];
} arg2T;

typedef union {
	__ea datatype* arg;
	char dummy[sizeof(__ea datatype*)];
} arg3T;

typedef union {
	int arg;
	char dummy[sizeof(int)];
} arg4T;


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
	for (i=0; i<sizeof(arg1T); ++i) {
		arg1.dummy[i] = stack[stack_used + i];
	}
	stack_used += sizeof(arg1T);

	arg2T arg2;
	for (i=0; i<sizeof(arg2T); ++i) {
		arg2.dummy[i] = stack[stack_used + i];
	}
	stack_used += sizeof(arg2T);

	arg3T arg3;
	for (i=0; i<sizeof(arg3T); ++i) {
		arg3.dummy[i] = stack[stack_used + i];
	}
	stack_used += sizeof(arg3T);

	arg4T arg4;
	for (i=0; i<sizeof(arg4T); ++i) {
		arg4.dummy[i] = stack[stack_used + i];
	}
	stack_used += sizeof(arg4T);


	last_block = 0;
	atomic_result = 0;

	for (blockIdx.x=start_calc; blockIdx.x < end_calc; ++blockIdx.x) {
	for (blockIdx.y=0; blockIdx.y<gridDim.y; ++blockIdx.y) {
	for (blockIdx.z=0; blockIdx.z<gridDim.z; ++blockIdx.z) {

		kernel (arg1.arg, arg2.arg, arg3.arg, arg4.arg);

	}
	}
	}


	__cache_flush();
	spu_write_out_mbox(0);
}

	return 0;
}
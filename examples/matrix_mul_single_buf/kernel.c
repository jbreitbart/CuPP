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

// for (i=0; i<16; ++i) {
// 	printf ("%d: ", i);
// 	for (j=0; j<16; ++j) {
// 		printf("%d ", B_local[i*16 + j]);
// 	}
// 	printf ("\n");
// }

datatype A_local[16*16] __attribute__ ((aligned (128)));
datatype B_local[16*16] __attribute__ ((aligned (128)));


static inline void kernel (addr64 A, addr64 B, __ea datatype* C, const int size) {
	struct dim3 threadIdx;

	local_datatype c_elem;
	

// 	const int block_nb = size / blockDim.x;
	int X=0;

	int tag = 0;
	int i=0;
	int j=0;


	// start transfer to first buffer
// 	mfc_get (A_local[0], A, blockDim.x*sizeof(datatype), tag, 0, 0);
// 	mfc_get (B_local[0], B, blockDim.y*sizeof(datatype), tag, 0, 0);



	for (X=0; X<512; ++X) {
		c_elem[X]=0;
	}

	for (X=0; X<size; X+=blockDim.x) {


	for (i=0; i<blockDim.x; ++i) {

// 		printf ("%d \n", i);
		mfc_get (&A_local[16*i],
		          A.ull + (X + size*(i + blockIdx.y*blockDim.y))*sizeof(datatype),
		          blockDim.x*sizeof(datatype), tag, 0, 0
		        );

// 		printf ("DMA A: %p, %llu, %d \n", &A_local[16*i], A.ull + (X + size*(i + blockIdx.y*blockDim.y))*sizeof(datatype), blockDim.x*sizeof(datatype));

		mfc_get (&B_local[16*i],
		          B.ull + (blockIdx.x*blockDim.x + size*(X + i))*sizeof(datatype),
		          blockDim.x*sizeof(datatype), tag, 0, 0
		        );

// 		printf ("DMA B: %p, %llu, %d \n", &B_local[16*i], B.ull + (blockIdx.x*blockDim.x + size*(X + i))*sizeof(datatype), blockDim.x*sizeof(datatype));
	}

	// complete transfer
	spu_writech (MFC_WrTagMask, 1 << tag);
	spu_mfcstat(MFC_TAG_UPDATE_ALL);
// 	tag ^= 1;

// 	printf(", \n");
	START


	for (i = 0; i < blockDim.x; i++) {
		LOCAL(c_elem) += A_local[i + blockDim.x*threadIdx.y] * B_local[threadIdx.x + blockDim.x*i];
//                                  A      [(size * thread_row) + i]           *B      [(size * i) + thread_col];
	}

	END

	}

	START

	const int thread_row = (blockDim.y * blockIdx.y) + threadIdx.y;
	const int thread_col = (blockDim.x * blockIdx.x) + threadIdx.x;

	C[(size * thread_row) + thread_col] = LOCAL(c_elem);

	END

}


typedef union {
	addr64 arg;
	char dummy[sizeof(addr64)];
} arg1T;

typedef union {
	addr64 arg;
	char dummy[sizeof(addr64)];
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
// 		printf (".\n");

		kernel (arg1.arg, arg2.arg, arg3.arg, arg4.arg);


	}
	}
	}


	__cache_flush();
	spu_write_out_mbox(0);
}

	return 0;
}

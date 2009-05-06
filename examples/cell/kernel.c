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

static inline void kernel (__ea int* arr) {
	struct dim3 threadIdx;

	__shared__ int shared[blockIdx.x];
	local_int temp;

	START

	LOCAL(temp) = 2;

	arr[threadIdx.x + blockIdx.x*blockDim.x] *= LOCAL(temp);

	END


}

int main () {
	// 1. we receive the block and grid size
	__ea struct dim3 *gridDim_ptr;
	gridDim_ptr = (__ea struct dim3*) spu_read_in_mbox();

	__ea struct dim3 *blockDim_ptr;
	blockDim_ptr  = (__ea struct dim3*) spu_read_in_mbox();

	gridDim  = *gridDim_ptr;
	blockDim = *blockDim_ptr;

	// 2. we receive all kernel parameters
	typedef __ea int* ea_int_ptr;
	__ea ea_int_ptr *arg1_ptr = (__ea ea_int_ptr*) spu_read_in_mbox();
	ea_int_ptr arg1 = *arg1_ptr;

	// 3. which blocks should we calculate
	const unsigned int start_calc = spu_read_in_mbox();
	const unsigned int end_calc = spu_read_in_mbox();


	for (blockIdx.x=start_calc; blockIdx.x < end_calc; ++blockIdx.x) {
	for (blockIdx.y=0; blockIdx.y<gridDim.y; ++blockIdx.y) {
	for (blockIdx.z=0; blockIdx.z<gridDim.z; ++blockIdx.z) {

		kernel (arg1);

	}
	}
	}

	__cache_flush();
	spu_write_out_mbox(0);

	return 0;
}

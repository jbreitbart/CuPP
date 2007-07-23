/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_cupp_runtime_H
#define CUPP_cupp_runtime_H

#if defined(__CUDACC__)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif

#include "cupp_common.h"

#include "exception/cuda_runtime_error.h"

#include <cuda_runtime.h>


namespace cupp {

template <typename T>
void mem_set(T* device_pointer, int value, const size_t size=1) {
	if (cudaMemset( reinterpret_cast<void*>( device_pointer ), value, sizeof(T)*size ) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
}


void* malloc_ (const unsigned int size_in_b) {
	void* temp;
	if (cudaMalloc( &temp, size_in_b ) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
	return temp;
}

template <typename T>
T* malloc(const size_t size=1) {
	return static_cast<T*> (malloc_ (size*sizeof(T)));
}


template <typename T>
void free(T* device_pointer) {
	if (cudaFree(device_pointer) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
}


template <typename T>
void copy_host_to_device(T *destination, const T * const source, size_t count=1) {
	if ( cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
}


template <typename T>
void copy_device_to_device(T* destination, const T * const source, size_t count=1) {
	if ( cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToDevice) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
}


template <typename T>
void copy_device_to_host(T* destination, const T * const source, size_t count=1) {
	if (cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
}


void thread_synchronize() {
	if (cudaThreadSynchronize() != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
}

} // namespace cupp

#endif

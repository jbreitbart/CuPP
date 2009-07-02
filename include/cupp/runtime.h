/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_runtime_H
#define CUPP_runtime_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif

#include <cstddef>
#include <libspe2.h>
#include <pthread.h>
#include <errno.h>
#include <iostream>


// CUPP
#include "cupp/common.h"
#include "cupp/exception/cell_runtime_error.h"

// CUDA
// #include <cuda_runtime.h>


namespace cupp {

template< typename T > class shared_device_pointer; // forward declaration to break cycle dependency

template <typename T>
void mem_set(T* device_pointer, int value, const size_t size=1);

template <typename T>
void mem_set(shared_device_pointer<T> device_pointer, int value, const size_t size=1);

inline void* malloc_ (const unsigned int size_in_b);

template <typename T>
T* malloc(const size_t size=1);

template <typename T>
void free(T* device_pointer);

template <typename T>
void copy_host_to_device(T *destination, const T * const source, size_t count=1);

template <typename T>
void copy_host_to_device(shared_device_pointer<T> destination, const T * const source, size_t count=1);

template <typename T>
void copy_device_to_device(T* destination, const T * const source, size_t count=1);


template <typename T>
void copy_device_to_device(shared_device_pointer<T> destination, const T * const source, size_t count=1);


template <typename T>
void copy_device_to_device(T *destination, const shared_device_pointer<T> source, size_t count=1);


template <typename T>
void copy_device_to_device(shared_device_pointer<T> destination, const shared_device_pointer<T> source, size_t count=1);


template <typename T>
void copy_device_to_host(T* destination, const T * const source, size_t count=1);


template <typename T>
void copy_device_to_host(T* destination, const shared_device_pointer<T> source, size_t count=1);

inline void thread_synchronize();

inline void put_in_mbox (spe_context_ptr_t spe, unsigned int *mbox_data, int count, unsigned int behavior);


inline void put_in_mbox (spe_context_ptr_t spe, unsigned int *mbox_data, int count, unsigned int behavior) {
	int status = 0;
	while (status == 0 || status== -1) {
		errno = 0;
		status = spe_in_mbox_write (spe, mbox_data, count, behavior);
		if (status!=0) {
			if (errno == ESRCH) {
				throw cupp::exception::cell_runtime_error  ("The specified SPE context is invalid.");
			}
			if (errno == EIO) {
				throw cupp::exception::cell_runtime_error  ("The I/O error occurred.");
			}
			if (errno == EINVAL) {
				throw cupp::exception::cell_runtime_error  ("The specified pointer to the mailbox message, the specified maximum number of mailbox entries, or the specified behavior is invalid.");
			}
		}
	}
}



// template <typename T>
// void mem_set(T* device_pointer, int value, const size_t size) {
// 	if (cudaMemset( reinterpret_cast<void*>( device_pointer ), value, sizeof(T)*size ) != cudaSuccess) {
// 		throw exception::cuda_runtime_error(cudaGetLastError());
// 	}
// }

// template <typename T>
// void mem_set(shared_device_pointer<T> device_pointer, int value, const size_t size) {
// 	mem_set(device_pointer.get(), value, size);
// }

inline void* malloc_ (const unsigned int size_in_b) {
	return new char[size_in_b];
// 	void* temp;
// 	if (cudaMalloc( &temp, size_in_b ) != cudaSuccess) {
// 		throw exception::cuda_runtime_error(cudaGetLastError());
// 	}
// 	return temp;
}

template <typename T>
T* malloc(const size_t size) {
	return static_cast<T*> (malloc_ (size*sizeof(T)));
}


template <typename T>
void free(T* device_pointer) {
	delete[] device_pointer;
// 	if (cudaFree(device_pointer) != cudaSuccess) {
// 		throw exception::cuda_runtime_error(cudaGetLastError());
// 	}
}


template <typename T>
void copy_host_to_device(T */*destination*/, const T * const /*source*/, size_t /*count*/) {
// 	if ( cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
// 		throw exception::cuda_runtime_error(cudaGetLastError());
// 	}
}


template <typename T>
void copy_host_to_device(shared_device_pointer<T> /*destination*/, const T * const /*source*/, size_t /*count*/) {
// 	copy_host_to_device(destination.get(), source, count);
}


template <typename T>
void copy_device_to_device(T* /*destination*/, const T * const /*source*/, size_t /*count*/) {
// 	if ( cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToDevice) != cudaSuccess) {
// 		throw exception::cuda_runtime_error(cudaGetLastError());
// 	}
}


template <typename T>
void copy_device_to_device(shared_device_pointer<T> /*destination*/, const T * const /*source*/, size_t /*count*/) {
// 	copy_device_to_device (destination.get(), source, count);
}


template <typename T>
void copy_device_to_device(T */*destination*/, const shared_device_pointer<T> /*source*/, size_t /*count*/) {
// 	copy_device_to_device (destination, source.get(), count);
}


template <typename T>
void copy_device_to_device(shared_device_pointer<T> /*destination*/, const shared_device_pointer<T> /*source*/, size_t /*count*/) {
// 	copy_device_to_device (destination.get(), source.get(), count);
}


template <typename T>
void copy_device_to_host(T* /*destination*/, const T * const /*source*/, size_t /*count*/) {
// 	if (cudaMemcpy(destination, source, count * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
// 		throw exception::cuda_runtime_error(cudaGetLastError());
// 	}
}


template <typename T>
void copy_device_to_host(T* /*destination*/, const shared_device_pointer<T> /*source*/, size_t /*count*/) {
// 	if (cudaMemcpy(destination, source.get(), count * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
// 		throw exception::cuda_runtime_error(cudaGetLastError());
// 	}
}

/**
 * Synchronizes the calling thread with the asynchronius CUDA calls. You should never need to call this manually
 */
inline void thread_synchronize() {
// 	if (cudaThreadSynchronize() != cudaSuccess) {
// 		throw exception::cuda_runtime_error(cudaGetLastError());
// 	}
}

} // namespace cupp

#endif

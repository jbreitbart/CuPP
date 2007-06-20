/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */


#include "device.h"

#include <string>

#include "exception/no_device.h"
#include "exception/no_supporting_device.h"
#include "exception/cuda_runtime_error.h"

#include <cuda_runtime.h>

namespace cupp {

device::device() {
	real_constructor(-1, -1, "Device Emulation");
}

device::device (const int major) {
	real_constructor(major, -1, 0);
}

device::device (const int major, const int minor) {
	real_constructor(major, minor, 0);
}

void device::real_constructor(const int major, const int minor, const char* name) {
	using namespace cupp::exception;

	// check if there is already a device
	int cur_device;
	if (cudaGetDevice(&cur_device)== cudaSuccess) {
		cudaSetDevice(cur_device);
		return;
	}

	const int device_cnt = 0;//device_count();

	if ( device_cnt == 0) {
		throw no_device();
	}

	int dev = 0;
	for (dev = 0; dev < device_cnt; ++dev) {
		cudaDeviceProp device_prop;
		cudaGetDeviceProperties(&device_prop, dev);
		bool take_it = false;

		//check major rev number
		if (name!=0) {
			if (std::string(name)==std::string(device_prop.name)) {
				take_it=true;
			} else {
				take_it=false;
			}
		}

		//check major rev number
		if (major!=-1) {
			if (major==device_prop.major) {
				take_it=true;
			} else {
				take_it=false;
			}
		}

		//check minor rev number
		if (minor!=-1) {
			if (minor==device_prop.minor) {
				take_it=true;
			} else {
				take_it=false;
			}
		}

		if ( take_it ) {
			break;
		}
	}

	if (dev == device_cnt) {
		throw no_supporting_device();
	}

	cudaSetDevice(dev);
}


}



#if 0
	public: /***  1D-MEMORY FUNCTIONS  ***/
		/**
		 * @brief Allocates 1-dimension memory on the device.
		 * @param size The number of elements you want to allocate on the device.
		 * @return A pointer to 1-dimension device memory.
		 * @exception cupp::exception::cuda_runtime_error Thrown if the memory allocation fails.
		 */
		template<typename T>
		memory1D<T> allocate_memory1D(std::size_t size) const {
			T* devptr;
			if (cudaMalloc( static_cast<void**>( &devptr ), sizeof(T)*size ) != cudaSuccess) {
				throw exception::cuda_runtime_error(cudaGetLastError());
			}

			return memory1D<T>(devptr, size);
		}

		/**
		 * @brief Frees memory on the GPU
		 * @param memory The memory which will be freeed
		 * @warning After this call is completed @a memory is no longer valid.
		 */
		template<typename T>
		void free (const memory1D<T>& memory) const {
			if (cudaFree(memory.pointer_) != cudaSuccess) {
				throw exception::cuda_runtime_error(cudaGetLastError());
			}
		}

		/**
		 * @brief Copies data from the host to the device
		 * @param src A pointer to the data we copy from
		 * @param dest A pointer to where the data will be copied
		 * @warning We will copy @a dest.size() many elements, you have to assure that @a src points to enough data.
		 */
		template <typename T>
		void copy_host_to_device(const T* src, const memory1D<T>& dest) {
			if (cudaMemcpy(dest.pointer_, src, dest.size() *sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
				throw exception::cuda_runtime_error(cudaGetLastError());
			}
		}

		/**
		 * @brief Copies data from the device to the host
		 * @param src A pointer to the data we copy from
		 * @param dest A pointer to where the data will be copied
		 * @warning We will copy @a src.size() many elements, you have to assure that @a dest points to enough data.
		 */
		template <typename T>
		void copy_device_to_host(const memory1D<T>& src, T* dest) {
			if (cudaMemcpy(dest, src.pointer_, p.size_*sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
				throw exception::cuda_runtime_error(cudaGetLastError());
			}
		}
		
	public: /*** UTILITY FUNCTIONS ***/
		/**
		 * @return the number of available devices
		 */
		static const int device_count() {
			int device_cnt = 0;
			cudaGetDeviceCount(&device_cnt);

			return device_cnt;
		}
#endif

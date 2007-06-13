/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_device_H
#define CUPP_device_H

#include "exception/no_device_ex.h"
#include "exception/no_supporting_device_ex.h"
#include "exception/cuda_runtime_ex.h"

#include <cstddef>
#include <string>
#include <cuda_runtime.h>

namespace cupp {

/**
 * @class device
 * @author Jens Breitbart
 * @version 0.1
 * @date 13.06.2007
 * @warning 
 * @brief asdvasd
 *
 * asdvasdvasdv
 */
class device {
// No device functionality when working with nvcc
#if !defined(__CUDACC__)
	public: /***  CONSTRUCTORS & DESTRUCTORS ***/
		/**
		 * @brief Generates a default device with no special requirements
		 */
		explicit device() {
			real_constructor(-1, -1, "Device Emulation");
		}

		/**
		 * @brief Generates a device with @a major revision number
		 * @param major The requested major revision number of the device
		 */
		explicit device (const int major) {
			real_constructor(major, -1, 0);
		}
		
		/**
		 * @brief Generates a device with @a major and @a minor revision number
		 * @param major The requested major revision number of the device
		 * @param minor The requested minor revision number of the device
		 */
		explicit device (const int major, const int minor) {
			real_constructor(major, minor, 0);
		}

	private:
		/**
		 * @brief This is the real constructor.
		 * @param major The requested major rev. number; pass -1 to ignore it
		 * @param minor The requested minor rev. number; pass -1 to ignore it
		 * @param name  The requested device name; pass 0 to ignore it
		 */
		void real_constructor(const int major, const int minor, const char* name) {
			// check if there is already a device
			int cur_device;
			if (cudaGetDevice(&cur_device)== cudaSuccess) {
				cudaSetDevice(cur_device);
				return;
			}
			
			const int device_count = device::number_of_devices();

			if (device_count == 0) {
				throw no_device_ex();
			}
			
			int dev;
			for (dev = 0; dev < device_count; ++dev) {
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
			
			if (dev == device_count) {
				throw no_supporting_device_ex();
			}
			
			cudaSetDevice(dev);
		}

	public: /***  1D-MEMORY FUNCTIONS  ***/
		/**
		 * @brief Allocates 1-dimension memory on the device.
		 * @param size The number of elements you want to allocate on the device.
		 * @return A pointer to 1-dimension device memory.
		 */
		template<typename T>
		device::memory1D<T> get_memory1D(std::size_t size) const {
			T* devptr;
			if (cudaMalloc((void**)&devptr, sizeof(T)*size) != cudaSuccess) {
				throw cuda_runtime_ex(cudaGetLastError());
			}

			return device::memory1D<T>(devptr, size);
		}

		/**
		 * @brief Frees memory on the GPU
		 * @param memory The memory which will be freeed
		 * @warning After this call is completed @a memory is no longer valid.
		 */
		template<typename T>
		void free (const device::memory1D<T> &memory) const {
			if (cudaFree(memory.pointer_) != cudaSuccess) {
				throw cuda_runtime_ex(cudaGetLastError());
			}
		}

		/**
		 * @brief Copies data from the host to the device
		 * @param src A pointer to the data we copy from
		 * @param dest A pointer to where the data will be copied
		 * @warning We will copy @a dest.size() many elements, you have to assure that @a src points to enough data.
		 */
		template <typename T>
		void copy_host_to_device(const T* src, const device::memory1D<T> &dest) {
			if (cudaMemcpy(p.pointer_, dest, p.size_*sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
				throw cuda_runtime_ex(cudaGetLastError());
			}
		}

		/**
		 * @brief Copies data from the device to the host
		 * @param src A pointer to the data we copy from
		 * @param dest A pointer to where the data will be copied
		 * @warning We will copy @a src.size() many elements, you have to assure that @a dest points to enough data.
		 */
		template <typename T>
		void copy_device_to_host(cons device::memory1D<T> &src, T* dest) {
			if (cudaMemcpy(dest, src.pointer_, p.size_*sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
				throw cuda_runtime_ex(cudaGetLastError());
			}
		}
		
	public: /*** UTILITY FUNCTIONS ***/
		/**
		 * @return the number of available devices
		 */
		static const int number_of_devices() {
			// set the device
			int device_count;
			cudaGetDeviceCount(&device_count);

			if (device_count == 0) {
				throw no_device_ex();
			}

			return device_count;
		}
#endif
};

/**
 * @class memory1D
 * @author Jens Breitbart
 * @version 0.1
 * @date 13.06.2007
 * @brief A pointer to linear 1-dimensional memory.
 * @warning You have to free this memory, or you will have a ressource leak!
 *
 * asdvasdvasdv
 */
template <typename T>
class device::memory1D {
	private: /***  INTERNAL DATA  ***/
		const T* pointer_;
		const std::size_t size_;
		
	private: /***  CONSTRUCTORS & DESTRUCTORS ***/
		memory1D() {};
		memory1D(const T* pointer, const std::size_t size) :
			pointer_(pointer), size_(size) {}
	public:
		/**
		 * @brief Copy constructor
		 * @param copy
		 * @todo Should we test if the pointer is used in the correct context?
		 * @todo Do we need a public copy-constructor?
		 */
		memory1D(const memory1D &copy) : pointer_(copy.pointer_), size_(copy.size_) {}

	public:
		
		/**
		 * @return How many elements we can store on the device
		 */
		std::size_t size() const {
			return size_;
		}

	public: /***  GPU FUNCTIONS  ***/
	#if defined(__CUDACC__)
		/**
		 * @brief Cast the memory1D into a simple pointer
		 * @warning Only available on the GPU.
		 */
		__device__
		operator T*() const {
			return pointer_;
		}
	#endif

	friend cupp::device;
};

}

#endif

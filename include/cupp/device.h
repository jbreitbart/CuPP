/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_device_H
#define CUPP_device_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif

#include <string>

// CUDA
#include <cuda_runtime.h>

#include "cupp/exception/no_device.h"
#include "cupp/exception/no_supporting_device.h"
#include "cupp/exception/cuda_runtime_error.h"
#include "cupp/runtime.h"


namespace cupp {


/**
 * @class device
 * @author Jens Breitbart
 * @version 0.2
 * @date 03.08.2007
 * @platform Host only!
 * @brief This class is a handle to a CUDA device. You need it to allocate data, run kernel, ...
 * @warning If you destroy your device object, all data located on it will be destroyed.
 */
class device {
	public:
		typedef int id_t;
	
	public: /***  CONSTRUCTORS & DESTRUCTORS ***/
		/**
		 * @brief Generates a default device with no special requirements
		 */
		device();

		/**
		 * @brief Generates a device with @a major revision number
		 * @param major The requested major revision number of the device
		 */
		explicit device (const int major);
		
		/**
		 * @brief Generates a device with @a major and @a minor revision number
		 * @param major The requested major revision number of the device
		 * @param minor The requested minor revision number of the device
		 */
		explicit device (const int major, const int minor);

		/**
		 * @brief Cleans up all ressources associated with the device
		 */
		~device();

	public:
		/**
		 * @brief This functions blocks until all requested tasks/kernels have been completed
		 */
		void sync() const;

		/**
		 * @return a unique id for @a this device
		 */
		int id() const;

	public: /***  GET INFORMATION ABOUT THE DEVICE  ***/
		/**
		 * @return ASCII string identifying this device
		 */
		const char* name() const { return device_prop_.name; }

		/**
		 * @return total amount of global memory available on this device in bytes
		 */
		size_t global_mem_size() const { return device_prop_.totalGlobalMem; }

		/**
		 * @return total amount of shared memory available on this device per block in bytes
		 */
		size_t shared_mem_size_per_block() const { return device_prop_.sharedMemPerBlock; }

		/**
		 * @return total number of registers available on this device per block
		 */
		int regs_per_block() const { return device_prop_.regsPerBlock; }

		/**
		 * @return warp size on this device
		 */
		int warp_size() const { return device_prop_.warpSize; }

		/**
		 * @return maximum memory pitch allowed on this device
		 */
		size_t mem_pitch() const { return device_prop_.memPitch; }

		/**
		 * @return maximum amount of threads per block
		 */
		int max_threads_per_block() const { return device_prop_.maxThreadsPerBlock; }

		/**
		 * @return maximum dimension of a thread block in each dimension
		 */
		int3 max_block_dimension() const { return make_int3(device_prop_.maxThreadsDim[0], device_prop_.maxThreadsDim[1], device_prop_.maxThreadsDim[2]); };

		/**
		 * @return maximum dimension of the grid in each dimension
		 */
		int3 max_grid_dimension() const { return make_int3(device_prop_.maxGridSize[0], device_prop_.maxGridSize[1], device_prop_.maxGridSize[2]); }

		/**
		 * @return total amount of constant memory on this device
		 */
		size_t constant_mem_size() const { return device_prop_.totalConstMem; }

		/**
		 * @return major revision number
		 */
		int major() const { return device_prop_.major; }

		/**
		 * @return minor revision number
		 */
		int minor() const { return device_prop_.minor; }

		/**
		 * @return clock frequency in kiloherz
		 */
		int clock_frequency() const { return device_prop_.clockRate; }

		/**
		 * @return alignment requirement of texture base addresses that does not require an offset to be applied to texture fetches
		 */
		size_t texture_alignment() const { return device_prop_.textureAlignment; }
	
	public: /***  UTILITY FUNCTIONS  ***/
		/**
		 * @return the number of available devices
		 */
		static int device_count();

	private:
		/**
		 * @brief This is the real constructor.
		 * @param major The requested major rev. number; pass -1 to ignore it
		 * @param minor The requested minor rev. number; pass -1 to ignore it
		 * @param name  The requested device name; pass 0 to ignore it
		 */
		void real_constructor(const int major, const int minor, const char* name);

	private:
		/**
		 * The properties of this device
		 */
		cudaDeviceProp device_prop_;

	private:
}; // class device


inline device::device() {
	real_constructor(1, -1, 0);
}

inline device::device (const int major) {
	real_constructor(major, -1, 0);
}

inline device::device (const int major, const int minor) {
	real_constructor(major, minor, 0);
}

inline device::~device() {
	cudaThreadExit();
}

inline void device::real_constructor(const int major, const int minor, const char* name) {
	using namespace cupp::exception;

	// check if there is already a device
	int cur_device;
	if (cudaGetDevice(&cur_device) == cudaSuccess) {
		cudaSetDevice(cur_device);
		cudaGetDeviceProperties(&device_prop_, cur_device);
		return;
		//throw too_many_devices_per_thread();
	}

	const int device_cnt = device_count();

	if ( device_cnt == 0) {
		throw no_device();
	}

	int dev = 0;
	for (dev = 0; dev < device_cnt; ++dev) {
		cudaGetDeviceProperties(&device_prop_, dev);
		bool take_it = false;

		//check major rev number
		if (name!=0) {
			if (std::string(name)==std::string(device_prop_.name)) {
				take_it=true;
			} else {
				take_it=false;
			}
		}

		//check major rev number
		if (major!=-1) {
			if (major<=device_prop_.major) {
				take_it=true;
			} else {
				take_it=false;
			}
		}

		//check minor rev number
		if (minor!=-1) {
			if (minor<=device_prop_.minor) {
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

inline void device::sync() const {
	cupp::thread_synchronize();
}

inline device::id_t device::id() const {
	int cur_device;
	cudaGetDevice(&cur_device);
	return cur_device;
}

inline int device::device_count() {
	int device_cnt = 0;
	cudaGetDeviceCount(&device_cnt);

	return device_cnt;
}

} // namespace cupp

#endif

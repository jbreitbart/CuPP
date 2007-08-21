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
#include "cupp_runtime.h"

namespace cupp {

device::device() {
	real_constructor(1, -1, 0);
}

device::device (const int major) {
	real_constructor(major, -1, 0);
}

device::device (const int major, const int minor) {
	real_constructor(major, minor, 0);
}

device::~device() {
	cudaThreadExit();
}

void device::real_constructor(const int major, const int minor, const char* name) {
	using namespace cupp::exception;

	// check if there is already a device
	int cur_device;
	if (cudaGetDevice(&cur_device)== cudaSuccess) {
		cudaSetDevice(cur_device);
		return;
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

void device::sync() {
	cupp::thread_synchronize();
}

int device::device_count() {
	int device_cnt = 0;
	cudaGetDeviceCount(&device_cnt);

	return device_cnt;
}


}

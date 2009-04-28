/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_EXCEPTION_too_many_devices_per_thread_H
#define CUPP_EXCEPTION_too_many_devices_per_thread_H


#if defined(__CUDACC__)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif


#include "cupp/exception/exception.h"

#include <cuda_runtime.h>

namespace cupp {
namespace exception {

/**
 * @class too_many_devices_per_thread
 * @author Jens Breitbart
 * @version 0.1
 * @date 23.08.2007
 * @brief This exception is thrown when you try to create a device in a thread that already has a device attached.
 */
class too_many_devices_per_thread : public exception {
	public:
		char const* what() const throw() {
			return "You created too many devices. You are currently limited to 1 device per thread";
		}
};

}
}

#endif

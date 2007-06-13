/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */


#ifndef CUPP_cuda_runtime_ex_H
#define CUPP_cuda_runtime_ex_H

#include "exception.h"

#include <cuda_runtime.h>

namespace cupp {

/**
 * @class cuda_runtime_ex
 * @author Jens Breitbart
 * @version 0.1
 * @date 13.06.2007
 * @brief This exception is thrown when an low-level CUDA error occurs, eg. an internal call to cudaMalloc fails.
 *
 * The returned error string is a default CUDA error string.
 */
class cuda_runtime_ex : cupp::exception {
	private:
		// the error
		cudaError_t error_;
	public:
		cuda_runtime_ex(cudaError_t error): error_(error) {}
		char const* what() const throw() {
			return cudaGetErrorString(error_);
		}
};

}

#endif

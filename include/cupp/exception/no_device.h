/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_no_device_H
#define CUPP_no_device_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif


#include "cupp/exception/exception.h"

namespace cupp {
namespace exception {

/**
 * @class no_device
 * @author Jens Breitbart
 * @version 0.1
 * @date 13.06.2007
 * @brief This exception is thrown when no CUDA device is found.
 */
class no_device : public exception {
	public:
		char const* what() const throw() {
			return "No CUDA device found.";
		}
};

} // namespace excpetion
} // namespace cupp

#endif

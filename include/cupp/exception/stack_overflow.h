/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_stack_overflow_H
#define CUPP_stack_overflow_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif


#include "cupp/exception/exception.h"

namespace cupp {
namespace exception {

/**
 * @class stack_overflow
 * @author Jens Breitbart
 * @version 0.1
 * @date 22.06.2007
 * @brief This exception is thrown when memory access non allocated memory is accessed
 */
class stack_overflow : public exception {
	public:
		char const* what() const throw() {
			return "You can only pass 256 bytes to a function call on the device.";
		}
};

} // namespace excpetion
} // namespace cupp

#endif

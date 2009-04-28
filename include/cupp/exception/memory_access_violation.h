/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_memory_access_violation_H
#define CUPP_memory_access_violation_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif


#include "cupp/exception/exception.h"

namespace cupp {
namespace exception {

/**
 * @class memory_access_violation
 * @author Jens Breitbart
 * @version 0.1
 * @date 21.06.2007
 * @brief This exception is thrown when non allocated memory is accessed
 */
class memory_access_violation : public exception {
	public:
		char const* what() const throw() {
			return "Memory access violation";
		}
};

} // namespace excpetion
} // namespace cupp

#endif

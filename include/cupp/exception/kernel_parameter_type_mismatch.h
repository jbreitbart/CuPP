/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_parameter_type_mismatch_H
#define CUPP_kernel_parameter_type_mismatch_H


#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif


#include "cupp/exception/exception.h"

namespace cupp {
namespace exception {

/**
 * @class kernel_parameter_type_mismatch
 * @author Jens Breitbart
 * @version 0.1
 * @date 23.07.2007
 * @brief This exception is thrown when a incompatible parameter is passed to a kernel
 */
class kernel_parameter_type_mismatch : public exception {
	public:
		/**
		 * @brief Generates an exception :-)
		 */
		kernel_parameter_type_mismatch() {}
		
		char const* what() const throw() {
			return "Wrong parameter type passed to kernel. Maybe forgot to set bindings?";
		}
};

}
}

#endif

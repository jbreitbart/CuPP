/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_number_of_parameters_mismatch_H
#define CUPP_kernel_number_of_parameters_mismatch_H


#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif


#include "cupp/exception/exception.h"

namespace cupp {
namespace exception {

/**
 * @class kernel_number_of_parameters_mismatch
 * @author Jens Breitbart
 * @version 0.1
 * @date 23.07.2007
 * @brief This exception is thrown when a wrong number of parameters is passed to a kernel.
 */
class kernel_number_of_parameters_mismatch : public exception {
	private:
		const int required_;
		const int passed_;
	public:
		/**
		 * @brief Generates an exception :-)
		 */
		kernel_number_of_parameters_mismatch(const int required, const int passed) : required_(required), passed_(passed) {}
		
		char const* what() const throw() {
			/// @todo maybe print out  required_, passed_ too
			return "Wrong number of parameter passed to a kernel.";
		}
};

}
}

#endif

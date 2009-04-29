/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_cell_runtime_error_H
#define CUPP_cell_runtime_error_H


#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif


#include "cupp/exception/exception.h"

namespace cupp {
namespace exception {

/**
 * @class cell_runtime_error
 * @author Jens Breitbart
 * @version 0.1
 * @date 28.04.2008
 * @brief This exception is thrown when an low-level CELL error occurs, eg. a thread could not be created
 */
class cell_runtime_error : public exception {
	private:
		// the error
		char* error_;
	public:
		/**
		 * @brief Generates an exception with the error number @a error
		 * @param error The error string
		 */
		cuda_runtime_error(char* error): error_(error) {}

		char const* what() const throw() {
			return error_;
		}
};

}
}

#endif

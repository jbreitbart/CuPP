/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#if defined(__CUDACC__)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif

#ifndef CUPP_no_supporting_device_H
#define CUPP_no_supporting_device_H

#include "exception.h"

namespace cupp {
namespace exception {

/**
 * @class no_supporting_device
 * @author Jens Breitbart
 * @version 0.1
 * @date 13.06.2007
 * @brief This exception is thrown when no CUDA device matches the requirements.
 */
class no_supporting_device : cupp::exception {
	public:
		char const* what() const throw() {
			return "No supported device found.";
		}
};

}
}

#endif

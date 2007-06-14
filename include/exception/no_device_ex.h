/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */


#ifndef CUPP_no_device_ex_H
#define CUPP_no_device_ex_H

#include "exception.h"

namespace cupp {

/**
 * @class no_device_ex
 * @author Jens Breitbart
 * @version 0.1
 * @date 13.06.2007
 * @brief This exception is thrown when no CUDA device is found.
 */
class no_device_ex : cupp::exception {
	public:
		char const* what() const throw() {
			return "No CUDA device found.";
		}
};

}

#endif

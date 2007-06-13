/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */


#ifndef CUPP_exception_H
#define CUPP_exception_H

#include <exception>

namespace cupp {

/**
 * @class exception
 * @author Jens Breitbart
 * @version 0.1
 * @date 13.06.2007
 * @note This class is pure virtual.
 * @brief The base class for all exceptions thrown by CUPP.
 *
 * It is based on the std::exception and can be handled the same way.
 */
class exception : std::exception {
	public:
		/**
		 * @return An error message in human readable form.
		 */
		virtual char const* what() const throw() = 0;
};

}

#endif

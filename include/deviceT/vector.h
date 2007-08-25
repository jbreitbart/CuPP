/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_DEVICET_vector_H
#define CUPP_DEVICET_vector_H

#include "cupp_common.h"
#include "kernel_type_binding.h"
#include "deviceT/memory1d.h"

namespace cupp {

template <typename T>
class vector;

namespace deviceT {

/**
 * @class vector
 * @author Jens Breitbart
 * @version 0.1
 * @date 24.07.2007
 * @platform Device only
 */

template< typename T >
class vector : public memory1d<T, cupp::vector<T> > {
	public:
		/**
		 * Set up the type bindings
		 */
		typedef vector<T>                                          device_type;
		typedef cupp::vector< typename get_type<T>::host_type >    host_type;
		
		/**
		 * @typedef size_type
		 * @brief The type you should use to index this class
		 */
		typedef typename memory1d<T, cupp::vector<T> >::size_type size_type;

		/**
		 * @typedef value_type
		 * @brief The type of data you want to store
		 */
		typedef typename memory1d<T, cupp::vector<T> >::value_type value_type;
};

} // namespace deviceT
} // namespace cupp

#endif

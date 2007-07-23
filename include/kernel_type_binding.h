/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_type_binding_H
#define CUPP_kernel_type_binding_H

#include <boost/type_traits.hpp>
#include <boost/any.hpp>

namespace cupp {

/**
 * @class kernel_type_binding
 * @author Jens Breitbart
 * @version 0.1
 * @date 22.07.2007
 * @brief These traits define the kernel type traits
 */

// this is the default trait for all types which require no special treatment
template <typename device_type>
class kernel_host_type {
	public:
		typedef device_type host_type;
};


// this is the default trait for all types which require no special treatment
template <typename host_type>
class kernel_device_type {
	public:
		typedef host_type device_type;
};


} // cupp

#endif //CUPP_kernel_call_traits_H

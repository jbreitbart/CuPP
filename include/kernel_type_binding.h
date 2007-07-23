/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_type_binding_H
#define CUPP_kernel_type_binding_H

#include <boost/type_traits.hpp>

namespace cupp {

/**
 * @class kernel_host_type
 * @author Jens Breitbart
 * @version 0.1
 * @date 22.07.2007
 * @brief These traits define the kernel parameter type-binding: device_type -> host_type
 * @note Read kernel_host_type<T>::type == give me the kernel host type for the kernel device type T
 */

// these are the default trait for all types which require no special treatment
// they will remove any cv qualified and any reference
// int -> int
template <typename device_type>
class kernel_host_type {
	public:
		typedef device_type type;
};

// T&  -> T
template <typename device_type>
class kernel_host_type<device_type&> {
	public:
		typedef typename kernel_host_type<typename boost::remove_reference<device_type>::type>::type type;
};

// const T& -> T&
template <typename device_type>
class kernel_host_type<device_type const> {
	public:
		typedef typename kernel_host_type<typename boost::remove_const<device_type>::type>::type type;
};

// volatile T -> T
template <typename device_type>
class kernel_host_type<device_type volatile> {
	public:
		typedef typename kernel_host_type<typename boost::remove_volatile<device_type>::type>::type type;
};


/**
 * @class kernel_device_type
 * @author Jens Breitbart
 * @version 0.1
 * @date 22.07.2007
 * @brief These traits define the kernel parameter type-binding: host_type -> device_type
 * @note Read kernel_device_type<T>::type == give me the kernel device type for the kernel host type T
 */

// these are the default trait for all types which require no special treatment
// they will remove any cv qualified and any reference
// int -> int
template <typename host_type>
class kernel_device_type {
	public:
		typedef host_type type;
};

// T&  -> T
template <typename host_type>
class kernel_device_type<host_type&> {
	public:
		typedef typename kernel_host_type<typename boost::remove_reference<host_type>::type>::type type;
};

// const T& -> T&
template <typename host_type>
class kernel_device_type<host_type const> {
	public:
		typedef typename kernel_host_type<typename boost::remove_const<host_type>::type>::type type;
};

// volatile T -> T
template <typename host_type>
class kernel_device_type<host_type volatile> {
	public:
		typedef typename kernel_host_type<typename boost::remove_volatile<host_type>::type>::type type;
};

} // cupp

#endif //CUPP_kernel_call_traits_H

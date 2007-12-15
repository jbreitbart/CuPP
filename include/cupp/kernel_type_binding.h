/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_type_binding_H
#define CUPP_kernel_type_binding_H

// not used with CUDA
#if !defined(NVCC)

#include <boost/type_traits.hpp>

#endif

namespace cupp {

namespace impl {

/**
 * @class is_pod
 * @author Jens Breitbart
 * @version 0.1
 * @date 15.12.2007
 * is_pod::value is true for all POD types
 */
template <typename T>
struct is_pod {
	enum {value = false};
};

#define MK_is_pod(T) \
template <> struct is_pod<T> { \
	enum {value = true}; \
};

MK_is_pod(void)

MK_is_pod(bool)
MK_is_pod(char)
MK_is_pod(signed char)
MK_is_pod(unsigned char)

MK_is_pod(signed short)
MK_is_pod(unsigned short)
MK_is_pod(signed int)
MK_is_pod(unsigned int)
MK_is_pod(signed long)
MK_is_pod(unsigned long)

MK_is_pod(float)
MK_is_pod(double)

/**
 * @class get_type_impl
 * @author Jens Breitbart
 * @version 0.1
 * @date 22.08.2007
 * Used by the @c get_type to get the type. There is a specialisation pod-types (where device_type == host_type).
 * This is the generic template.
 */
template <bool POD, typename T>
struct get_type {
	typedef typename T::host_type    host_type;
	typedef typename T::device_type  device_type;
};

/**
 * Specialisation for pod-types.
 */
template <typename T>
struct get_type<true, T> {
	typedef T    host_type;
	typedef T    device_type;
};

}

/**
 * @class get_type
 * @author Jens Breitbart
 * @version 0.3
 * @date 15.12.2007
 * This can be used to get the host or device type a template. POD-types have T as there device and there host type.
 */
template <typename T>
struct get_type {
	typedef typename impl::get_type <impl::is_pod < T >::value, T>::host_type      host_type;
	typedef typename impl::get_type <impl::is_pod < T >::value, T>::device_type    device_type;
};


#if !defined(NVCC)

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
		typedef typename get_type<device_type>::host_type type;
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
template <typename host_type>
struct kernel_device_type {
	typedef typename get_type<host_type>::device_type type;
};

// T&  -> T
template <typename host_type>
struct kernel_device_type<host_type&> {
	typedef typename kernel_device_type<typename boost::remove_reference<host_type>::type>::type type;
};

// const T& -> T&
template <typename host_type>
struct kernel_device_type<host_type const> {
	typedef typename kernel_device_type<typename boost::remove_const<host_type>::type>::type type;
};

// volatile T -> T
template <typename host_type>
struct kernel_device_type<host_type volatile> {
	typedef typename kernel_device_type<typename boost::remove_volatile<host_type>::type>::type type;
};

#endif


} // cupp

#endif //CUPP_kernel_call_traits_H

/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_type_binding_H
#define CUPP_kernel_type_binding_H

// not used with CUDA
#if !defined(__CUDACC__)

#include <boost/type_traits.hpp>

#endif

namespace cupp {

namespace impl {

/**
 * @class has_typdefs
 * @author Jens Breitbart
 * @version 0.1
 * @date 10.02.2008
 * Used by the @c cupp::get_type to detect wether @c T has @c T::host_type and @c T::device_type typedefs or not.
 */
template <typename T>
class has_typdefs {
	private:
		typedef char one;
		typedef char (&two)[2];
		
		template<typename R> struct helper;

		template<class S> static one check(helper<typename S::host_type>*, helper<typename S::device_type>*);
		template<class S> static two check(...);

	public:
		enum {value = (sizeof(check<T>(0, 0)) == sizeof(char))};
};


/**
 * @class get_type
 * @author Jens Breitbart
 * @version 0.2
 * @date 10.02.2008
 * Used by the @c cupp::get_type to get the type. There is a specialisation for types not offering @c T::host_type or @c T::device_type.
 */
template <bool POD, typename T>
struct get_type {
	typedef typename T::host_type    host_type;
	typedef typename T::device_type  device_type;
};

/**
 * Specialisation for types without the host/device type typedefs..
 */
template <typename T>
struct get_type<false, T> {
	typedef T    host_type;
	typedef T    device_type;
};

} //cupp::impl

/**
 * @class get_type
 * @author Jens Breitbart
 * @version 0.4
 * @date 10.02.2008
 * This can be used to get the host or device type a template. POD-types have T as there device and there host type.
 */
template <typename T>
struct get_type {
	typedef typename impl::get_type <impl::has_typdefs < T >::value, T>::host_type      host_type;
	typedef typename impl::get_type <impl::has_typdefs < T >::value, T>::device_type    device_type;
};

/**
 * specialization for the CUDA types
 * @todo fix this in a more clean way
 */


#define SET_TYPE_SPEC(a) \
template <> \
struct get_type<a> { \
	typedef a      host_type; \
	typedef a    device_type; \
};

SET_TYPE_SPEC(dim3)
SET_TYPE_SPEC(char1)
SET_TYPE_SPEC(char2)
SET_TYPE_SPEC(char3)
SET_TYPE_SPEC(char4)
SET_TYPE_SPEC(uchar1)
SET_TYPE_SPEC(uchar2)
SET_TYPE_SPEC(uchar3)
SET_TYPE_SPEC(uchar4)

SET_TYPE_SPEC(short1)
SET_TYPE_SPEC(short2)
SET_TYPE_SPEC(short3)
SET_TYPE_SPEC(short4)
SET_TYPE_SPEC(ushort1)
SET_TYPE_SPEC(ushort2)
SET_TYPE_SPEC(ushort3)
SET_TYPE_SPEC(ushort4)

SET_TYPE_SPEC(int1)
SET_TYPE_SPEC(int2)
SET_TYPE_SPEC(int3)
SET_TYPE_SPEC(int4)
SET_TYPE_SPEC(uint1)
SET_TYPE_SPEC(uint2)
SET_TYPE_SPEC(uint3)
SET_TYPE_SPEC(uint4)


SET_TYPE_SPEC(long1)
SET_TYPE_SPEC(long2)
SET_TYPE_SPEC(long3)
SET_TYPE_SPEC(long4)
SET_TYPE_SPEC(ulong1)
SET_TYPE_SPEC(ulong2)
SET_TYPE_SPEC(ulong3)
SET_TYPE_SPEC(ulong4)

SET_TYPE_SPEC(float1)
SET_TYPE_SPEC(float2)
SET_TYPE_SPEC(float3)
SET_TYPE_SPEC(float4)

SET_TYPE_SPEC(double1)
SET_TYPE_SPEC(double2)

#undef SET_TYPE_SPEC

#if !defined(__CUDACC__)

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


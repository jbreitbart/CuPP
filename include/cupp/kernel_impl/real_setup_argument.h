/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_real_setup_argument_H
#define CUPP_KERNEL_IMPL_real_setup_argument_H

#include "cupp/common.h"

#include <boost/type_traits.hpp>
#include <boost/any.hpp>

namespace cupp {
namespace kernel_impl {

/**
 * @class real_setup_argument
 * @author Jens Breitbart
 * @version 0.2
 * @date 13.02.2008
 * @brief Use by cupp::kernel_impl::kernel_launcher_impl to introspec the __global__ function and call a function inside the kernel_launcher_impl.
 * @warning The is very strictly tied to kernel_launcher_impl; I would not expect any usage beyond this.
 * @note This is a friend class of kernel_launcher_impl, but not defined into the same header to make it easier to extend it and to keep it clear.
 */

template <int arity>
class real_setup_argument {

	/**
	 * @brief This will call the function that.setup_argument with the correct template parameters and @a arg as parameter. The correct template parameter is based on the template argument passed to kernel_launcher_impl (see example for a more insightful understanding :-)).
	 @example To be more precise: If T is kernel_launcher_impl<void (*) (int, double)> and @a pos = 2 than the function that.setup_arument<double>(arg) is called.
	 * @param T is expected to be of type kernel_launcher_impl< whatever >
	 */
	template <typename T>
	static boost::any set (const device &d, const boost::any &arg, const int pos, T &that);

};

#define CUPP_REAL_SETUP_ARGUMENT_HEADER(a,b) \
template <> \
class real_setup_argument<a> { \
	template <typename T> \
	inline static boost::any set (const device &d, const boost::any &arg, const int pos, T &that); \
	\
	template <typename T> \
	friend class kernel_launcher_impl; \
	\
	template <typename T> \
	friend boost::any real_setup_argument<b>::set(const device &d, const boost::any&, const int, T&); \
};

template <>
class real_setup_argument<16> {
	template <typename T>
	inline static boost::any set (const device &d, const boost::any &arg, const int pos, T &that);
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	//template <typename T>
	//friend boost::any real_setup_argument<17>::set(const device &d, const boost::any&, const int, T&);
};

//CUPP_REAL_SETUP_ARGUMENT_HEADER(16,17)
CUPP_REAL_SETUP_ARGUMENT_HEADER(15,16)
CUPP_REAL_SETUP_ARGUMENT_HEADER(14,15)
CUPP_REAL_SETUP_ARGUMENT_HEADER(13,14)
CUPP_REAL_SETUP_ARGUMENT_HEADER(12,13)
CUPP_REAL_SETUP_ARGUMENT_HEADER(11,12)
CUPP_REAL_SETUP_ARGUMENT_HEADER(10,11)
CUPP_REAL_SETUP_ARGUMENT_HEADER(9,10)
CUPP_REAL_SETUP_ARGUMENT_HEADER(8,9)
CUPP_REAL_SETUP_ARGUMENT_HEADER(7,8)
CUPP_REAL_SETUP_ARGUMENT_HEADER(6,7)
CUPP_REAL_SETUP_ARGUMENT_HEADER(5,6)
CUPP_REAL_SETUP_ARGUMENT_HEADER(4,5)
CUPP_REAL_SETUP_ARGUMENT_HEADER(3,4)
CUPP_REAL_SETUP_ARGUMENT_HEADER(2,3)
CUPP_REAL_SETUP_ARGUMENT_HEADER(1,2)
CUPP_REAL_SETUP_ARGUMENT_HEADER(0,1)


#undef CUPP_REAL_SETUP_ARGUMENT_HEADER


/*** IMPLEMENTATION ***/

template <typename T>
boost::any real_setup_argument<0>::set (const device &d, const boost::any &arg, const int pos, T &that) {
	UNUSED_PARAMETER(d);
	UNUSED_PARAMETER(arg);
	UNUSED_PARAMETER(that);
	UNUSED_PARAMETER(pos);
	return boost::any();
}


#define CUPP_REAL_SETUP_ARGUMENT(a,b) \
template <typename T> \
boost::any real_setup_argument<a>::set (const device &d, const boost::any &arg, const int pos, T &that) { \
	if (pos == a) { \
		typedef typename boost::function_traits <typename T::F> :: arg##a##_type ARG; \
		return that.template setup_argument<ARG> (d, arg); \
	} \
	return real_setup_argument<b>::set(d, arg, pos, that); \
}

CUPP_REAL_SETUP_ARGUMENT(1,0)
CUPP_REAL_SETUP_ARGUMENT(2,1)
CUPP_REAL_SETUP_ARGUMENT(3,2)
CUPP_REAL_SETUP_ARGUMENT(4,3)
CUPP_REAL_SETUP_ARGUMENT(5,4)
CUPP_REAL_SETUP_ARGUMENT(6,5)
CUPP_REAL_SETUP_ARGUMENT(7,6)
CUPP_REAL_SETUP_ARGUMENT(8,7)
CUPP_REAL_SETUP_ARGUMENT(9,8)
CUPP_REAL_SETUP_ARGUMENT(10,9)
CUPP_REAL_SETUP_ARGUMENT(11,10)
CUPP_REAL_SETUP_ARGUMENT(12,11)
CUPP_REAL_SETUP_ARGUMENT(13,12)
CUPP_REAL_SETUP_ARGUMENT(14,13)
CUPP_REAL_SETUP_ARGUMENT(15,14)
CUPP_REAL_SETUP_ARGUMENT(16,15)


#undef CUPP_REAL_SETUP_ARGUMENT

} // kernel_impl
} // cupp

#endif //CUPP_KERNEL_IMPL_real_setup_argument_H

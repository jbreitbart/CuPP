/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_real_setup_argument_H
#define CUPP_KERNEL_IMPL_real_setup_argument_H

#include <boost/type_traits.hpp>
#include <boost/any.hpp>

namespace cupp {
namespace kernel_impl {

/**
 * @class real_setup_argument
 * @author Björn Knafla Some very useful writting of the word "template". :-)
 * @author Jens Breitbart
 * @version 0.1
 * @date 21.07.2007
 * @brief Use by cupp::kernel_impl::kernel_launcher_impl to introspec the __global__ function and call a function inside the kernel_launcher_impl.
 * @warning The is very strictly tidied to kernel_launcher_impl; I would not expect any usage beyond this.
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
	static boost::any set (const boost::any &arg, const int pos, T &that);
};

template <>
class real_setup_argument<0> {
	template <typename T>
	static boost::any set (const boost::any &arg, const int pos, T &that) { return boost::any(); }
};

template <>
class real_setup_argument<1> {
	template <typename T>
	static boost::any set (const boost::any &arg, const int pos, T &that) {
		if (pos == 1) {
			typedef typename boost::function_traits <typename T::F> :: arg1_type ARG;
			return that.template setup_argument<ARG> (arg);
		}
	}
};

template <>
class real_setup_argument<2> {
	template <typename T>
	static boost::any set (const boost::any &arg, const int pos, T &that) {
		if (pos == 2) {
			typedef typename boost::function_traits <typename T::F> :: arg2_type ARG;
			return that.template setup_argument<ARG> (arg);
		}
		return real_setup_argument<1>::set(arg, pos, that);
	}
};

template <>
class real_setup_argument<3> {
	template <typename T>
	static boost::any set (const boost::any &arg, const int pos, T &that) {
		if (pos == 3) {
			typedef typename boost::function_traits <typename T::F> :: arg3_type ARG;
			return that.template setup_argument<ARG> (arg);
		}
		return real_setup_argument<2>::set(arg, pos, that);
	}
};


} // kernel_impl
} // cupp

#endif //CUPP_KERNEL_IMPL_real_setup_argument_H

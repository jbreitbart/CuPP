/* The following code example is taken from the book
 * "C++ Templates - The Complete Guide"
 * by David Vandevoorde and Nicolai M. Josuttis, Addison-Wesley, 2002
 *
 * (C) Copyright David Vandevoorde and Nicolai M. Josuttis 2002.
 * Permission to copy, use, modify, sell and distribute this software
 * is granted provided this copyright notice appears in all copies.
 * This software is provided "as is" without express or implied
 * warranty, and with no claim as to its suitability for any purpose.
 */
#ifndef CUPP_KERNEL_MAGIC_forward_param_H
#define CUPP_KERNEL_MAGIC_forward_param_H


namespace cupp {
namespace kernel_magic {

/**
 * @class forward_param
 * @author David Vandevoorde and Nicolai M. Josuttis in the book "C++ Templates - The Complete Guide"
 * @author Jens Breitbart modified the original template to better fit our requirements
 * @version 0.1
 * @date 23.06.2007
 * @brief Used by cupp::kernel to pass parameters
 *
 * @a forward_param<T>::type is
 * - constant reference for all types
 * - a dummy type for type void
 */
template<typename T>
class forward_param {
	public:
		typedef T const & type;
};

template<>
class forward_param<void> {
	private:
		class unused {};
	public:
		typedef unused type;
};

}
}

#endif // CUPP_KERNEL_MAGIC_forward_param_H

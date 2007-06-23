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

#ifndef CUPP_KERNEL_MAGIC_function_ptr_H
#define CUPP_KERNEL_MAGIC_function_ptr_H

namespace cupp {
namespace kernel_magic {

/**
 * @class forward_param
 * @author David Vandevoorde and Nicolai M. Josuttis in the book "C++ Templates - The Complete Guide"
 * @author Jens Breitbart modified the original template to better fit our requirements
 * @version 0.1
 * @date 23.06.2007
 * @brief Used by cupp::kernel to determine the parameters of the function pointer
 */

// primary template handles maximum number of parameters:
template<typename P1 = void,
         typename P2 = void,
         typename P3 = void>
class function_ptr {
	public:
		typedef void (*type)(P1,P2,P3);
};

// partial specialization for two parameters:
template<typename P1,
         typename P2>
class function_ptr<P1, P2, void> {
	public:
		typedef void (*type)(P1,P2);
};

// partial specialization for one parameter:
template<typename P1>
class function_ptr<P1, void, void> {
	public:
		typedef void (*type)(P1);
};

// partial specialization for no parameters:
template<>
class function_ptr<void, void, void> {
	public:
		typedef void (*type)();
};

}
}

#endif //CUPP_KERNEL_MAGIC_function_ptr_H

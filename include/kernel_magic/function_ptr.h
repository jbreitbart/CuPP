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

#ifndef CUPP_KERNEL_IMPL_function_ptr_H
#define CUPP_KERNEL_IMPL_function_ptr_H

namespace cupp {
namespace kernel_impl {

/**
 * @class function_ptr
 * @author David Vandevoorde and Nicolai M. Josuttis in the book "C++ Templates - The Complete Guide"
 * @author Jens Breitbart modified the original template to better fit our requirements
 * @version 0.2
 * @date 19.07.2007
 * @brief Used by cupp::kernel to determine the parameters of the function pointer
 */

template<typename P1 = void,
         typename P2 = void,
         typename P3 = void>
class function_ptr {
	public:
		enum { num_params=3 };
		typedef void (*type)(P1,P2,P3);
		typedef P1 ParamT1;
		typedef P2 ParamT2;
		typedef P3 ParamT3;
};

// partial specialization for two parameters:
template<typename P1,
         typename P2>
class function_ptr<P1, P2, void> {
	public:
		enum { num_params=2 };
		typedef void (*type)(P1,P2);
		typedef P1 ParamT1;
		typedef P2 ParamT2;
		typedef void ParamT3;
};

// partial specialization for one parameter:
template<typename P1>
class function_ptr<P1, void, void> {
	public:
		enum { num_params=1 };
		typedef void (*type)(P1);
		typedef P1 ParamT1;
		typedef void ParamT2;
		typedef void ParamT3;
};

// partial specialization for no parameters:
template<>
class function_ptr<void, void, void> {
	public:
		enum { num_params=0 };
		typedef void (*type)();
		typedef void ParamT1;
		typedef void ParamT2;
		typedef void ParamT3;
};



// helper functions, which deduct the template parameters on there own :-)
template <typename P1, typename P2, typename P3> inline
function_ptr<P1, P2, P3> make_fptr( void (*fp)(P1, P2, P3) ) {
	return function_ptr<P1, P2, P3>();
}

template <typename P1, typename P2> inline 
function_ptr<P1, P2> make_fptr( void (*fp)(P1, P2) ) {
	return function_ptr<P1, P2>();
}

template <typename P1> inline 
function_ptr<P1> make_fptr( void (*fp)(P1) ) {
	return function_ptr<P1>();
}

inline
function_ptr<void, void, void> make_fptr( void (*fp)() ) {
	return function_ptr<void, void, void>();
}

}
}

#endif //CUPP_KERNEL_IMPL_function_ptr_H

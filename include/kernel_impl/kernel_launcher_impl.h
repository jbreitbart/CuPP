/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_kernel_launcher_impl_H
#define CUPP_KERNEL_IMPL_kernel_launcher_impl_H

#include "kernel_launcher_base.h"
#include "function_ptr.h"
#include "is_second_level_const.h"

#include <boost/any.hpp>
#include <deque>

namespace cupp {
namespace kernel_impl {

/**
 * @class kernel_launcher_impl
 * @author Björn Knafla: Initial design
 * @author Jens Breitbart
 * @version 0.1
 * @date 19.07.2007
 * @brief 
 */

template< class F >
class kernel_launcher_impl : public kernel_launcher_base {
	public:
		kernel_launcher_impl (F a) : func_(a) {};

		virtual void setup_argument( const boost::any &arg, const int pos );

		virtual void launch() {
			//return cudaLaunch( func_ );
		}

		virtual std::deque<bool> dirty_parameters() const {
			return test_dirty (make_fptr(func_));
		}
		
		virtual int number_of_param() const {
			return real_number_of_param (make_fptr(func_));
		}


		
	private:
		template <typename T1, typename T2, typename T3>
		static std::deque<bool> test_dirty (const function_ptr<T1, T2, T3> &fp);

		template <typename T1, typename T2>
		static std::deque<bool> test_dirty (const function_ptr<T1, T2> &fp);

		template <typename T1>
		static std::deque<bool> test_dirty (const function_ptr<T1> &fp);

		static std::deque<bool> test_dirty (const function_ptr<> &fp);
		
		template <typename T>
		static int real_number_of_param(const T& a) {
			return T::num_params;
		}

	private:
		F func_;
		size_t offset_;
};


template< class F >
template< typename T1, typename T2, typename T3 >
std::deque<bool> kernel_launcher_impl<F>::test_dirty (const function_ptr<T1, T2, T3> &fp) {

	std::deque<bool> qu;

	qu.push_back ( !is_second_level_const< T1 >::value );
	qu.push_back ( !is_second_level_const< T2 >::value );
	qu.push_back ( !is_second_level_const< T3 >::value );

	return qu;
}

template< class F >
template< typename T1, typename T2 >
std::deque<bool> kernel_launcher_impl<F>::test_dirty (const function_ptr<T1, T2> &fp) {

	std::deque<bool> qu;

	qu.push_back ( !is_second_level_const< T1 >::value );
	qu.push_back ( !is_second_level_const< T2 >::value );

	return qu;
}

template< class F >
template< typename T1 >
std::deque<bool> kernel_launcher_impl<F>::test_dirty (const function_ptr<T1> &fp) {

	std::deque<bool> qu;

	qu.push_back ( !is_second_level_const< T1 >::value );

	return qu;
}

template< class F >
std::deque<bool> kernel_launcher_impl<F>::test_dirty (const function_ptr<> &fp) {
	return std::deque<bool>();
}

} // kernel_impl
} // cupp

#endif //CUPP_KERNEL_IMPL_is_second_level_const_H

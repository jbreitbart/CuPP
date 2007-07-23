/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_kernel_launcher_impl_H
#define CUPP_KERNEL_IMPL_kernel_launcher_impl_H

// CUPP
#include "kernel_impl/kernel_launcher_base.h"
#include "kernel_impl/is_second_level_const.h"
#include "kernel_impl/real_setup_argument.h"
#include "kernel_impl/test_dirty.h"

#include "exception/cuda_runtime_error.h"
#include "exception/stack_overflow.h"
#include "exception/kernel_parameter_type_mismatch.h"

#include "kernel_call_traits.h"
#include "kernel_type_binding.h"
#include "cupp_runtime.h"

// STD
#include <vector>
#include <iostream>

// BOOST
#include <boost/type_traits.hpp>
#include <boost/any.hpp>

namespace cupp {
namespace kernel_impl {

/**
 * @class kernel_launcher_impl
 * @author Björn Knafla: Initial design and some enlightening comments.
 * @author Jens Breitbart
 * @version 0.2
 * @date 21.07.2007
 * @brief Use by cupp::kernel to push arguments on the cuda function stack, call the __global__ cuda function.
 */

template< typename F_ >
class kernel_launcher_impl : public kernel_launcher_base {
	public:
		
		/**
		 * @typedef F
		 * @brief The function type of the __global__ cuda function
		 */
		typedef typename boost::remove_pointer<F_>::type F;

		/**
		 * This is the arity of the function.
		 * @example F = void (*)(int, int) => arity == 2
		 * @example F = void (*)(void)     => arity == 0
		 */
		enum { arity = boost::function_traits<F>::arity };
		
		
		/**
		 * @brief Constructor
		 * @param a A pointer to the __global__ cuda function to be called
		 * @param grid_dim The dimension and size of the grid
		 * @param block_dim The dimension and size of the block
		 * @param shared_mem The amount of dynamic shared memory needed by the kernel
		 * @param tokens The number of tokens
		 */
		kernel_launcher_impl (F func, const dim3 &grid_dim, const dim3 &block_dim, const size_t shared_mem=0, const int tokens = 0) : func_(func), grid_dim_(grid_dim), block_dim_(block_dim), shared_mem_(shared_mem), tokens_(tokens), stack_in_use_(0) {};


		/**
		 * Configures the cuda launch.
		 */
		virtual void configure_call();

		
		/**
		 * @brief Checks if the type of the passed @a arg matches the parameter @a pos of the __global__ cuda function. If everything is ok it will check the @a push_on_function_stack trait, call some possible defined conversation and than pushes the parameter on the cuda function stack.
		 * @param arg A pointer to the to be pushed argument
		 * @param pos The position to which the passed @a arg matches to the parameter of the __global__ function. (NOTE: first position is 1 not 0, we follow the boost naming here!)
		 * @return a pointer to the created device_copy of type kernel_type_binding<>::device_type
		 */
		virtual boost::any setup_argument( const boost::any &arg, const int pos ) {
			return real_setup_argument< arity >::set (arg, pos, *this);
		}


		/**
		 * @brief Calls the __global__ function.
		 */
		virtual void launch();


		/**
		 * @brief Checks which parameters could be changed by @a launch()
		 * @return A vector with the size of arity. True at position 0 means the data which has been passed to the first parameter of the function could have been changed by the function call and should be marked dirty.
		 */
		virtual std::vector<bool> dirty_parameters() const {
			return test_dirty< arity >::template dirty< F >();
		}


	private:
		/**
		 * @brief Doing the real work for the public-virtual-non-template version of this function
		 * @param T The type of what is expected inside @a arg. May be different type if and only if the wrong type is passed to cupp::kernel::operator().
		 */
		template <typename T>
		boost::any setup_argument (const boost::any &arg);

		/**
		 * @brief Put parameter @a a on the execution stack of the kernel
		 * @param a The parameter to be copied on the stack
		 */
		template <typename T>
		void put_argument_on_stack(const T &a);

	private:
		/**
		 * A pointer to the __global__ cuda function.
		 */
		F* func_;

		/**
		 * Grid dimension
		 */
		dim3 grid_dim_;

		/**
		 * Block dimension
		 */
		dim3 block_dim_;

		/**
		 * The size of the shared memory
		 */
		size_t shared_mem_;

		/**
		 * The tokens ... whatever this may be, cuda docu ist kind of unspecific here
		 */
		int tokens_;
		
		/**
		 * How many cuda function call stack space is currently in use.
		 */
		size_t stack_in_use_;

		template <int i>
		friend class real_setup_argument;
};


template< typename F_ >
void kernel_launcher_impl<F_>::configure_call() {
	if (cudaConfigureCall(grid_dim_, block_dim_, shared_mem_, tokens_) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
}


template< typename F_ >
void kernel_launcher_impl<F_>::launch() {
	if (cudaLaunch((const char*)func_) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
	
	stack_in_use_ = 0;
}


template< typename F_ >
template <typename T>
boost::any kernel_launcher_impl<F_>::setup_argument (const boost::any &arg) {
	using namespace boost;
	
	typedef typename kernel_device_type<T>::type device_type;

	// get the host type matching our device_type
	typedef typename kernel_host_type<device_type>::type host_type;

	//get what is inside our any
	host_type* temp = 0;

	try {
		temp = const_cast <host_type*> (any_cast< const host_type* > (arg));
	} catch (boost::bad_any_cast &e) {
		// ok, something is wrong with the types
		// let's throw our own exception here
		throw exception::kernel_parameter_type_mismatch();
	}

	// get our copy to be passed to the device
	/// @TODO can this be a reference?
	const device_type &device_copy = kernel_call_traits<host_type, device_type>::get_device_copy(*temp);

	if (is_reference <T>()) {
		using namespace std;
		// ok this means our kernel wants a reference

		// copy device_copy into global memory
		device_type* device_copy_ptr = cupp::malloc<device_type>();
		cupp::copy_host_to_device(device_copy_ptr, &device_copy);

		// push address of device_copy in global memory of type device_type* on kernel_stack
		put_argument_on_stack(device_copy_ptr);
		
		// return address of of type add_pointer<device_type>
		return boost::any(device_copy_ptr);
	} else {
		// push device_type auf kernel stack
		put_argument_on_stack(device_copy);

		// return an empty any, this should trigger when some will try to cast it
		return boost::any();
	}
}


template< typename F_ >
template <typename T>
void kernel_launcher_impl<F_>::put_argument_on_stack(const T &a) {
	if (stack_in_use_+sizeof(T) > 256) {
		throw exception::stack_overflow();
	}
	if (cudaSetupArgument(&a, sizeof(T), stack_in_use_) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
	// std::cout << a << std::endl;
	// std::cout << stack_in_use_ << " => " << stack_in_use_ + sizeof(T) << std::endl;
	stack_in_use_ += sizeof(T);
}

} // kernel_impl
} // cupp

#endif //CUPP_KERNEL_IMPL_is_second_level_const_H

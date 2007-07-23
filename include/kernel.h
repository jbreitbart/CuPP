/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_H
#define CUPP_kernel_H

#if defined(__CUDACC__)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif


// CUPP
#include "exception/kernel_number_of_parameters_mismatch.h"
#include "kernel_impl/kernel_launcher_base.h"
#include "kernel_impl/kernel_launcher_impl.h"
#include "kernel_type_binding.h"
#include "kernel_call_traits.h"
#include "device.h"

// STD
#include <vector>

// BOOST
#include <boost/any.hpp>
#include <boost/type_traits.hpp>

// CUDA
#include <vector_types.h>


namespace cupp {

using namespace cupp::kernel_impl;

/**
 * @class kernel
 * @author Björn Knafla: Initial design and some enlightening comments.
 * @author Jens Breitbart
 * @version 0.2
 * @date 21.06.2007
 * @platform Host only!
 * @brief asdvasd
 *
 * asdvasdvasdv
 */

class kernel {
	public:
		/**
		 * @brief Constructor used to generate a kernel
		 * @param f A pointer to the kernel function
		 * @param grid_dim The dimension of the grid, the kernel we be executed on
		 * @param block_dim The dimension of the block, the kernel we be executed on
		 * @param shared_mem The number of dynamic shared memory needed by this kernel (in bytes)
		 * @param tokens
		 */
		template< typename CudaKernelFunc>
		kernel( CudaKernelFunc f, const dim3 &grid_dim, const dim3 &block_dim, const size_t shared_mem=0, const int tokens = 0) : number_of_parameters_(boost::function_traits < typename boost::remove_pointer<CudaKernelFunc>::type >::arity) {
		
			kb_ = new kernel_launcher_impl< CudaKernelFunc >(f, grid_dim, block_dim, shared_mem, tokens);
			dirty = kb_->dirty_parameters();
		}

		/**
		 * @brief Just our destructor
		 */
		~kernel() {  delete kb_;  }


		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 */
		void operator()(const device &d);


		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 */
		template< typename P1 >
		void operator()(const device &d, const P1 &p1 );
		
		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 */
		template< typename P1, typename P2 >
		void operator()(const device &d, const P1 &p1, const P2 &p2 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 */
		template< typename P1, typename P2, typename P3 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3 );

		
	private:
		/**
		 * @brief Calls the dirty kernel_call_traits function if needed
		 * @param p The parameter passed when the kernel
		 * @param i The number of the parameter (1 == first parameter)
		 */
		template <typename P>
		inline void handle_call_traits(const P &p, const int i);

		/**
		 * @brief Checks if @a number matches with @a number_of_parameters_
		 * @param number The number to check with
		 */
		inline void check_number_of_parameters (const int number);

	private:
		/**
		 * @brief The arity of our function
		 */
		const int number_of_parameters_;

		/**
		 * @brief Our internal kernel_launcher ... he does all the work :-)
		 */
		kernel_launcher_base* kb_;

		/**
		 * @brief Stores if a parameter is passed be non-const reference to our __global__. If yes we need to call kernel_call_traits::dirty
		 */
		std::vector<bool> dirty;

		/**
		 * @brief Stores the valuse returned by kb_ -> setup_argument(). They are needed by ther kernel_call_traits.
		 */
		std::vector<boost::any> returnee_vec;
};


void kernel::check_number_of_parameters (const int number) {
	if (number_of_parameters_ != number) {
		throw exception::kernel_number_of_parameters_mismatch(number_of_parameters_, number);
	}
}


template <typename P>
void kernel::handle_call_traits(const P &p, const int i) {
	if (dirty[i-1]) {
		typedef typename kernel_device_type<P>::type device_type;
		typedef typename kernel_device_type<P>::type host_type;
		device_type *device_ptr = boost::any_cast<device_type *>(returnee_vec[i-1]);

		kernel_call_traits<device_type, host_type>::dirty(p, device_ptr);
	}
}


/***  OPERATPR()  ***/
void kernel::operator()(const device &d) {
	check_number_of_parameters(0);
	
	kb_ -> configure_call();

	kb_->launch();
}

template< typename P1 >
void kernel::operator()(const device &d, const P1 &p1 ) {
	check_number_of_parameters(1);
	
	kb_ -> configure_call();

	returnee_vec.push_back ( kb_-> setup_argument( boost::any(&p1), 1 ) );

	kb_->launch();

	handle_call_traits (p1, 1);

	returnee_vec.clear();
}

template< typename P1, typename P2 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2 ) {
	check_number_of_parameters(2);
	
	kb_ -> configure_call();

	returnee_vec.push_back ( kb_-> setup_argument( boost::any(&p1), 1 ) );
	returnee_vec.push_back ( kb_-> setup_argument( boost::any(&p2), 2 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);

	returnee_vec.clear();
}

template< typename P1, typename P2, typename P3 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3 ) {
	check_number_of_parameters(3);
	
	kb_ -> configure_call();

	returnee_vec.push_back ( kb_-> setup_argument( boost::any(&p1), 1 ) );
	returnee_vec.push_back ( kb_-> setup_argument( boost::any(&p2), 2 ) );
	returnee_vec.push_back ( kb_-> setup_argument( boost::any(&p3), 3 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p2, 3);

	returnee_vec.clear();
}








































#if 0






/// @todo wie dokumentiert man eigentlich template parameter
/// @todo a grid is only 2 dimensional!
template <typename P1 = void,
          typename P2 = void,
          typename P3 = void>
class kernel {
	private:
		/**
		 * @brief The type of function executed on the device
		 */
		typedef typename function_ptr<P1, P2, P3>::type kernel_pointer;
	
	public:
		/**
		 * @brief Generates a kernel which executes the function @a f on a device
		 * @param f A pointer to the kernel you want to execute on the device
		 */
		kernel(kernel_pointer f);

		/**
		 * @brief Generates a kernel which executes the function @a f on a device
		 * @param f A pointer to the kernel you want to execute on the device
		 * @param grid_dim The dimension and size of the grid
		 * @param block_dim The dimension and size of the block
		 */
		kernel(kernel_pointer p, dim3 grid_dim, dim3 block_dim);


		/**
		 * @brief Sets the dimension of the grid for all following calls to @a grid_dim
		 * @param grid_dim The new grid dimension
		 */
		void set_grid_dim(dim3 grid_dim);

		/**
		 * @brief Sets the dimension of the block for all following calls to @a block_dim
		 * @param block_dim The new block dimension
		 */
		void set_block_dim(dim3 block_dim);


		/**
		 * @brief Executes the kernel
		 * @param d The device on which the kernel will be executed
		 */
		void operator()(const device &d);

		/**
		 * @brief Executes the kernel
		 * @param d The device on which the kernel will be executed
		 * @param a1 The first kernel parameter
		 */
		void operator()(const device &d, typename forward_param<P1>::type a1);

		/**
		 * @brief Executes the kernel
		 * @param d The device on which the kernel will be executed
		 * @param a1 The first kernel parameter
		 * @param a2 The second kernel parameter
		 */
		void operator()(const device &d, typename forward_param<P1>::type a1, typename forward_param<P2>::type a2);

		/**
		 * @brief Executes the kernel
		 * @param d The device on which the kernel will be executed
		 * @param a1 The first kernel parameter
		 * @param a2 The second kernel parameter
		 * @param a3 The third kernel parameter
		 */
		void operator()(const device &d, typename forward_param<P1>::type a1, typename forward_param<P2>::type a2, typename forward_param<P3>::type a3);

		virtual void operator()(...){};

	private:
		/**
		 * @brief Put parameter @a a on the execution stack of the kernel
		 * @param a The parameter to be copied on the stack
		 */
		template <typename T>
		void put_argument_on_stack(const T &a);
	
		/**
		 * @brief Specifies the grid/block size for the next call
		 * @warning This must be called before you call put_argument_on_stack
		 */
		void configure_call();
		
	private:
		/**
		 * The pointer to the function executed on the device
		 */
		kernel_pointer kernel_;

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
		 * @todo add it to constructors
		 */
		size_t shared_mem_size_;
		
		/**
		 * How much of the stack is already filled
		 */
		size_t stack_used_;
};


template <typename P1, typename P2, typename P3>
kernel<P1, P2, P3>::kernel(kernel_pointer f) : kernel_(f), grid_dim_(1), block_dim_(1), stack_used_(0) {
}


template <typename P1, typename P2, typename P3>
kernel<P1, P2, P3>::kernel(kernel_pointer f, dim3 grid_dim, dim3 block_dim) : kernel_(f), grid_dim_(grid_dim),  block_dim_(block_dim), stack_used_(0) {
}


template <typename P1, typename P2, typename P3>
void kernel<P1, P2, P3>::operator()(const device &d) {
	typename function_ptr<void, void, void>::type temp = kernel_;

	configure_call();
	
	if (cudaLaunch((const char*)temp) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	};

	stack_used_ = 0;
}


template <typename P1, typename P2, typename P3>
void kernel<P1, P2, P3>::operator()(const device &d, typename forward_param<P1>::type a1) {
	typename function_ptr<P1, void, void>::type temp = kernel_;

	configure_call();

	put_argument_on_stack(a1);

	if (cudaLaunch((const char*)temp) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	};

	stack_used_ = 0;
}


template <typename P1, typename P2, typename P3>
void kernel<P1, P2, P3>::operator()(const device &d, typename forward_param<P1>::type a1, typename forward_param<P2>::type a2) {
	typename function_ptr<P1, P2, void>::type temp = kernel_;

	configure_call();

	put_argument_on_stack(a1);
	put_argument_on_stack(a2);

	if (cudaLaunch((const char*)temp) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	};

	stack_used_ = 0;
}


template <typename P1, typename P2, typename P3>
void kernel<P1, P2, P3>::operator()(const device &d, typename forward_param<P1>::type a1, typename forward_param<P2>::type a2, typename forward_param<P3>::type a3) {
	typename function_ptr<P1, P2, P3>::type temp = kernel_;

	configure_call();

	put_argument_on_stack(a1);
	put_argument_on_stack(a2);
	put_argument_on_stack(a3);

	if (cudaLaunch((const char*)temp) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	};

	stack_used_ = 0;
}


template <typename P1, typename P2, typename P3>
template <typename T>
void kernel<P1, P2, P3>::put_argument_on_stack(const T &a) {
	if (stack_used_+sizeof(T) > 256) {
		throw exception::stack_overflow();
	}
	if (cudaSetupArgument(&a, sizeof(T), stack_used_) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
	stack_used_+=sizeof(T);
}


template <typename P1, typename P2, typename P3>
void kernel<P1, P2, P3>::configure_call() {
	if (cudaConfigureCall(grid_dim_, block_dim_) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
}

#endif

}

#endif

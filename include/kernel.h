/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_H
#define CUPP_kernel_H

#if defined(__CUDACC__)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif

#include "device.h"

#include <vector_types.h>



#include "kernel_impl/kernel_launcher_base.h"
#include "kernel_impl/kernel_launcher_impl.h"

#include "kernel_type_binding.h"
#include "kernel_call_traits.h"

#include <vector>

#include <boost/any.hpp>


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
		template< typename CudaKernelFunc>
		kernel( CudaKernelFunc f, const dim3 &grid_dim, const dim3 &block_dim, const size_t shared_mem=0, const int tokens = 0) : number_of_parameters_(boost::function_traits < CudaKernelFunc >::arity) {
			kb_ = new kernel_launcher_impl< CudaKernelFunc >(f, grid_dim, block_dim, shared_mem, tokens);
		}

		~kernel() {
			delete kb_;
		}

		template< typename P1, typename P2 >
		void operator()( const P1 &p1, const P2 &p2 ) {
		
			if (number_of_parameters_ != 2) {
				/// @todo throw exception
			}

			kb_ -> configure_call();

			std::vector<boost::any> returnee_vec;
			returnee_vec.push_back ( kb_-> setup_argument( &p1, 1 ) );
			returnee_vec.push_back ( kb_-> setup_argument( &p2, 2 ) );

			kb_->launch();
			
			const std::vector<bool> dirty = kb_->dirty_parameters();

			if (dirty[0]) {
				typename kernel_device_type<P1>::type *device_ptr = boost::any_cast<typename kernel_device_type<P1>::type *>(returnee_vec[0]);

				kernel_call_traits<P1, typename kernel_device_type<P1>::type>::dirty(p1, device_ptr);
			}

			if (dirty[1]) {
				typename kernel_device_type<P2>::type *device_ptr = boost::any_cast<typename kernel_device_type<P2>::type *>(returnee_vec[1]);

				kernel_call_traits<P1, typename kernel_device_type<P2>::type>::dirty(p2, device_ptr);
			}
		}

	private:
		const int number_of_parameters_;
		kernel_launcher_base* kb_;
};











































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

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
#include "kernel_magic/forwardparam.hpp"
#include "kernel_magic/functionptrt.hpp"
#include "exception/cuda_runtime_error.h"
#include "exception/stack_overflow.h"

#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>

namespace cupp {

using namespace cupp::kernel_magic;

/**
 * @class kernel
 * @author Jens Breitbart
 * @version 0.1
 * @date 21.06.2007
 * @platform Host only!
 * @brief asdvasd
 *
 * asdvasdvasdv
 */

/// @code_review wie dokumentiert man eigentlich template parameter
/// @todo a grid is only 2 dimensional!
template <typename P1 = void,
          typename P2 = void,
          typename P3 = void>
class kernel {
	private:
		/**
		 * @brief The type of function executed on the device
		 */
		typedef typename FunctionPtrT<void, P1, P2, P3>::Type kernel_pointer;
	
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
		void operator()(const device &d, const P1 &a1 /*typename ForwardParamT<P1>::Type a1*/);

		/**
		 * @brief Executes the kernel
		 * @param d The device on which the kernel will be executed
		 * @param a1 The first kernel parameter
		 * @param a2 The second kernel parameter
		 */
		void operator()(const device &d, typename ForwardParamT<P1>::Type a1, typename ForwardParamT<P2>::Type a2);

		/**
		 * @brief Executes the kernel
		 * @param d The device on which the kernel will be executed
		 * @param a1 The first kernel parameter
		 * @param a2 The second kernel parameter
		 * @param a3 The third kernel parameter
		 */
		void operator()(const device &d, typename ForwardParamT<P1>::Type a1, typename ForwardParamT<P2>::Type a2, typename ForwardParamT<P3>::Type a3);

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
		 * The pointer to function executed on the device
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
	typename FunctionPtrT<void, void, void, void>::Type temp = kernel_;

	configure_call();
	
	if (cudaLaunch((const char*)temp) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	};

	stack_used_ = 0;
}


template <typename P1, typename P2, typename P3>
void kernel<P1, P2, P3>::operator()(const device &d, const P1 &a1 /*typename ForwardParamT<P1>::Type a1*/) {
	typename FunctionPtrT<void, P1, void, void>::Type temp = kernel_;
	configure_call();

	put_argument_on_stack(a1);

	if (cudaLaunch((const char*)temp) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	};

	stack_used_ = 0;
}


template <typename P1, typename P2, typename P3>
void kernel<P1, P2, P3>::operator()(const device &d, typename ForwardParamT<P1>::Type a1, typename ForwardParamT<P2>::Type a2) {
	typename FunctionPtrT<void, P1, P2, void>::Type temp = kernel_;

	configure_call();

	put_argument_on_stack(a1);
	put_argument_on_stack(a2);

	if (cudaLaunch((const char*)temp) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	};

	stack_used_ = 0;
}


template <typename P1, typename P2, typename P3>
void kernel<P1, P2, P3>::operator()(const device &d, typename ForwardParamT<P1>::Type a1, typename ForwardParamT<P2>::Type a2, typename ForwardParamT<P3>::Type a3) {
	typename FunctionPtrT<void, P1, P2, P3>::Type temp = kernel_;

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

}

#endif

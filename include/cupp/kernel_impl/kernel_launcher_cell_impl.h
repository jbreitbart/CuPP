/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_kernel_launcher_cell_impl_H
#define CUPP_KERNEL_IMPL_kernel_launcher_cell_impl_H

// CUPP
#include "cupp/kernel_impl/kernel_launcher_base.h"
#include "cupp/kernel_impl/is_second_level_const.h"
#include "cupp/kernel_impl/real_setup_argument.h"
#include "cupp/kernel_impl/test_dirty.h"

#include "cupp/exception/cell_runtime_error.h"
#include "cupp/exception/stack_overflow.h"
#include "cupp/exception/kernel_parameter_type_mismatch.h"

#include "cupp/kernel_call_traits.h"
#include "cupp/kernel_type_binding.h"
#include "cupp/runtime.h"
// #include "cupp/shared_device_pointer.h"
// #include "cupp/device_reference.h"



// CUDA
#include "cupp/cell/cuda_stub.h"

// Cell
#include <libspe2.h>
#include <pthread.h>
#include <errno.h>


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
 * @author Jens Breitbart
 * @version 0.1
 * @date 28.04.2008
 * @brief Use by cupp::kernel to call the kernel on the Cell
 */

template< typename F_ >
class kernel_launcher_cell_impl : public kernel_launcher_base {
	public:
		
		/**
		 * @typedef F
		 * @brief The function type of the kernel function
		 */
		typedef typename boost::remove_pointer<F_>::type F;

		/**
		 * This is the arity of the function.
		 * @example F_ = void (*)(int, int) => arity == 2
		 * @example F_ = void (*)(void)     => arity == 0
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
		kernel_launcher_cell_impl (F /*func*/, spe_program_handle_t *prog_handle, const dim3 &grid_dim, const dim3 &block_dim, const size_t shared_mem=0, const int tokens = 0) :
		prog_handle_(prog_handle), grid_dim_(grid_dim), block_dim_(block_dim), shared_mem_(shared_mem), tokens_(tokens), stack_in_use_(0) {
			stack_ = new char[2*sizeof(dim3) + 256];
		}

		~kernel_launcher_cell_impl() {
			delete[] stack_;
		}


		/**
		 * Configures the cuda launch. Specifies the grid/block size for the next call.
		 */
		virtual void configure_call(const device& d);

		
		/**
		 * @brief Checks if the type of @a arg matches the parameter @a pos of the __global__ cuda function. If everything is ok it will put it on the __global__ function stack
		 * @param arg A pointer to the to be pushed argument
		 * @param pos The position to which the passed @a arg matches to the parameter of the __global__ function. (NOTE: first position is 1 not 0, we follow the boost naming here!)
		 * @return a pointer to the created device_copy of type kernel_type_binding<>::device_type
		 * @warning You must call configure_call() before you call this function!
		 */
		virtual boost::any setup_argument(const device &d, const boost::any &arg, const int pos ) {
			return real_setup_argument< arity >::set (d, arg, pos, *this);
		}


		/**
		 * @brief Calls the __global__ function.
		 */
		virtual void launch(const device&);


		/**
		 * @brief Checks which parameters could be changed by @a launch()
		 * @return A vector with the size of arity. True at position 0 means the data which has been passed to the first parameter of the function could have been changed by the function call and should be marked dirty. ~ 1 only if a parameter is passed as reference
		 */
		static std::vector<bool> dirty_parameters() {
			return test_dirty< boost::function_traits<F>::arity >::template dirty< F >();
		}

		/**
		 * @brief Change the grid dimension
		 */
		virtual void set_grid_dim ( const dim3& grid_dim ) { grid_dim_ = grid_dim; }

		/**
		 * @return The current grid dimension
		 */
		virtual dim3 grid_dim ( ) { return grid_dim_; }
		
		/**
		 * @brief Change the block dimension
		 */
		virtual void set_block_dim ( const dim3& block_dim ) { block_dim_ = block_dim; }

		/**
		 * @return The current block dimension
		 */
		virtual dim3 block_dim  ( ) { return block_dim_; }
		
		/**
		 * @brief Change the size of the dynamic shared memory
		 */
		virtual void set_shared_mem ( const size_t& shared_mem ) { shared_mem_ = shared_mem; }

		/**
		 * @return The current size of dynamic shared memory
		 */
		virtual size_t shared_mem ( ) { return shared_mem_; }

	private:
		/**
		 * @brief Doing the real work for the public-virtual-non-template version of this function
		 * @param T The type of what is expected inside @a arg. May be different type if and only if the wrong type is passed to cupp::kernel::operator().
		 */
		template <typename T>
		boost::any setup_argument (const device &d, const boost::any &arg);

		/**
		 * @brief Put parameter @a a on the execution stack of the kernel
		 * @param a The parameter to be copied on the stack
		 */
		template <typename T>
		void put_argument_on_stack(const device &d, const T &a);

	private:
		/**
		 * The spe program handle
		 */
		spe_program_handle_t *prog_handle_;

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

		/**
		 * The stack
		 */
		char *stack_;

		/**
		 * The SPE contextes
		 */
		spe_context_ptr_t *ctxs_;

		/**
		 * The pthreads used to run the spes
		 */
		pthread_t *threads_;

		template <int i>
		friend class real_setup_argument;
};


static inline void *spe_kernel_pthread_function(void* arg) {
	spe_context_ptr_t ctx;

	unsigned int entry = SPE_DEFAULT_ENTRY;
	ctx = *((spe_context_ptr_t *)arg);

	if (spe_context_run(ctx, &entry, 0, NULL, NULL, NULL) < 0) {
		throw cupp::exception::cell_runtime_error ("Failed running context");
	}

	pthread_exit(NULL);
	return 0;
}


template< typename F_ >
void kernel_launcher_cell_impl <F_>::configure_call(const device& d) {

	// start the SPE threads
	ctxs_ = new spe_context_ptr_t[d.spes()];
	threads_ = new pthread_t[d.spes()];

	for (int i=0; i<d.spes(); ++i) {
		// Create context
		if ((ctxs_[i] = spe_context_create (0, NULL)) == NULL) {
			throw cupp::exception::cell_runtime_error ("Failed creating context");
		}

		// Load program into context
		if (spe_program_load (ctxs_[i], prog_handle_)) {
			throw cupp::exception::cell_runtime_error ("Failed loading program");
		}

		// Create thread for each SPE context
		if (pthread_create (&threads_[i], NULL, &spe_kernel_pthread_function, &ctxs_[i]))  {
			throw cupp::exception::cell_runtime_error ("Failed creating thread");
		}
	}

	memcpy (stack_, (char*)&grid_dim_, sizeof(dim3));
	memcpy (stack_+sizeof(dim3), (char*)&block_dim_, sizeof(dim3));

/*	// send grid / block dim to the SPE
	for (int i=0; i<d.spes(); ++i) {
		unsigned int buffer = reinterpret_cast<unsigned int> (&grid_dim_);
		put_in_mbox (ctxs_[i], &buffer, 1, SPE_MBOX_ALL_BLOCKING);
	}

	for (int i=0; i<d.spes(); ++i) {
		unsigned int buffer = reinterpret_cast<unsigned int> (&block_dim_);
		put_in_mbox (ctxs_[i], &buffer, 1, SPE_MBOX_ALL_BLOCKING);
	}*/
}


template< typename F_ >
void kernel_launcher_cell_impl <F_>::launch(const device& d) {

	// static work scheduling

	const int number_of_work_per_spe = grid_dim_.x / d.spes();

	unsigned int start = 0;
	unsigned int end = number_of_work_per_spe;

	char* stack_ptr = stack_;
	unsigned int buffer = (unsigned int)stack_ptr;

	for (int i=0; i<d.spes(); ++i) {
		put_in_mbox (ctxs_[i], &buffer, 1, SPE_MBOX_ALL_BLOCKING);
		put_in_mbox (ctxs_[i], &start, 1, SPE_MBOX_ALL_BLOCKING);
		put_in_mbox (ctxs_[i], &end, 1, SPE_MBOX_ALL_BLOCKING);
		
		start += number_of_work_per_spe;
		end = (i!=d.spes()-2) ? (end+number_of_work_per_spe) : grid_dim_.x;
	}

// 	stack_in_use_ = 0;

	// wait until all spes have finished their work
	for (int i=0; i<d.spes(); ++i) {
		unsigned int data = 1;
		while (spe_out_mbox_status(ctxs_[i]) != 1) {}
		spe_out_mbox_read(ctxs_[i], &data, 1);
		if (data != 0) {
			throw cupp::exception::cell_runtime_error ("Something is completly wrong here ... aka \"the error that should not happen nb. 1a\"");
		}
	}

	for (int i=0; i<d.spes(); ++i) {
		if (pthread_join (threads_[i], NULL)) {
			throw cupp::exception::cell_runtime_error ("Failed pthread_join");
		}
	
		/* Destroy context */
		if (spe_context_destroy (ctxs_[i]) != 0) {
			throw cupp::exception::cell_runtime_error ("Failed destroying context");
		}
	}


	delete[] ctxs_;
	delete[] threads_;
}


template< typename F_ >
template <typename T>
boost::any kernel_launcher_cell_impl <F_>::setup_argument (const device &d, const boost::any &arg) {

	using namespace boost;
	
	// get the host type matching our device_type
	typedef typename kernel_host_type<T>::type host_type;
	typedef typename kernel_device_type<host_type>::type device_type;

	//get what is inside our any
	host_type* temp = 0;

	try {
		temp = const_cast <host_type*> (any_cast< const host_type* > (arg));
	} catch (boost::bad_any_cast &e) {
		// ok, something is wrong with the types
		// let's throw our own exception here
		throw exception::kernel_parameter_type_mismatch();
	}
	
	//invoke the copy constructor ...
	//host_type host_copy (*temp);
	
	//const device_type device_copy = kernel_call_traits<host_type, device_type>::transform(d, host_copy);
	
	// push device_type auf kernel stack
	put_argument_on_stack(d, *temp);

	// return an empty any, this should trigger when some tries to cast it
	return boost::any();
}


template< typename F_ >
template <typename T>
void kernel_launcher_cell_impl <F_>::put_argument_on_stack(const device &/*d*/, const T &a) {
	if (stack_in_use_+sizeof(T) > 256) {
		throw exception::stack_overflow();
	}

	int pos = 2*sizeof(dim3) + stack_in_use_;
	memcpy (stack_+pos, &a, sizeof(T));

// 	for (int i=0; i<d.spes(); ++i) {
// 		unsigned int buffer = reinterpret_cast<unsigned int> (&a);
// 		put_in_mbox (ctxs_[i], &buffer, 1, SPE_MBOX_ALL_BLOCKING);
// 	}


	stack_in_use_ += sizeof(T);
}

} // kernel_impl
} // cupp

#endif //CUPP_KERNEL_IMPL_is_second_level_const_H

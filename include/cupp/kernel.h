/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_H
#define CUPP_kernel_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif


// CUPP
#include "cupp/exception/kernel_number_of_parameters_mismatch.h"
#include "cupp/kernel_impl/kernel_launcher_base.h"
#include "cupp/kernel_impl/kernel_launcher_impl.h"
#include "cupp/kernel_type_binding.h"
#include "cupp/kernel_call_traits.h"
#include "cupp/device.h"
#include "cupp/device_reference.h"

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
 * @author Björn Knafla: Initial design
 * @author Jens Breitbart
 * @version 0.3
 * @date 03.08.2007
 * @platform Host only!
 * @brief Represents a __global__ function
 */

class kernel {
	public:
		/**
		 * @brief Constructor used to generate a kernel
		 * @param f A pointer to the kernel function
		 * @param shared_mem The number of dynamic shared memory needed by this kernel (in bytes)
		 * @param tokens
		 */
		template< typename CudaKernelFunc>
		kernel( CudaKernelFunc f, const size_t shared_mem=0, const int tokens = 0) :
		number_of_parameters_ ( boost::function_traits < typename boost::remove_pointer<CudaKernelFunc>::type >::arity ),
		dirty ( kernel_launcher_impl< CudaKernelFunc >::dirty_parameters() ) {

			dim3 grid_dim;
			dim3 block_dim;
			kb_ = new kernel_launcher_impl< CudaKernelFunc >(f, grid_dim, block_dim, shared_mem, tokens);
		}
		
		/**
		 * @brief Constructor used to generate a kernel
		 * @param f A pointer to the kernel function
		 * @param grid_dim The dimension of the grid, the kernel we be executed on
		 * @param block_dim The dimension of the block, the kernel we be executed on
		 * @param shared_mem The number of dynamic shared memory needed by this kernel (in bytes)
		 * @param tokens
		 */
		template< typename CudaKernelFunc>
		kernel( CudaKernelFunc f, const dim3 &grid_dim, const dim3 &block_dim, const size_t shared_mem=0, const int tokens = 0) :
		number_of_parameters_(boost::function_traits < typename boost::remove_pointer<CudaKernelFunc>::type >::arity),
		dirty ( kernel_launcher_impl< CudaKernelFunc >::dirty_parameters() ) {
		
			kb_ = new kernel_launcher_impl< CudaKernelFunc >(f, grid_dim, block_dim, shared_mem, tokens);
		}

		/**
		 * @brief Just our destructor
		 */
		~kernel() {  delete kb_;  }


		/**
		 * @brief Change the grid dimension
		 */
		void set_grid_dim ( const dim3& grid_dim ) { kb_ -> set_grid_dim (grid_dim); }

		/**
		 * @return The current grid dimension
		 */
		dim3 grid_dim ( ) { return kb_ -> grid_dim(); }
		
		/**
		 * @brief Change the block dimension
		 */
		void set_block_dim ( const dim3& block_dim ) { kb_ -> set_block_dim (block_dim); }

		/**
		 * @return The current block dimension
		 */
		dim3 block_dim  ( ) { return kb_ -> block_dim(); }
		
		/**
		 * @brief Change the size of the dynamic shared memory
		 */
		void set_shared_mem ( const size_t& shared_mem ) { kb_ -> set_shared_mem (shared_mem); }

		/**
		 * @return The current size of dynamic shared memory
		 */
		size_t shared_mem ( ) { return kb_ -> shared_mem(); }
		
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

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5 );
		
		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6 );
		
		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 * @param p8 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 * @param p8 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 * @param p8 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 * @param p8 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 * @param p8 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 * @param p8 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12, typename P13 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12, const P13 &p13 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 * @param p8 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12, typename P13, typename P14 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12, const P13 &p13, const P14 &p14 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 * @param p8 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12, typename P13, typename P14, typename P15 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12, const P13 &p13, const P14 &p14, const P15 &p15 );

		/**
		 * @brief Calls the kernel.
		 * @param d The device where you want the kernel to be executed on
		 * @param p1 The first parameter to be passed to the kernel
		 * @param p2 The second parameter to be passed to the kernel
		 * @param p3 The third parameter to be passed to the kernel
		 * @param p4 ...
		 * @param p5 ...
		 * @param p6 ...
		 * @param p7 ...
		 * @param p8 ...
		 */
		template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12, typename P13, typename P14, typename P15, typename P16 >
		void operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12, const P13 &p13, const P14 &p14, const P15 &p15, const P16 &p16 );


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
		const std::vector<bool> dirty;

		/**
		 * @brief Stores the valuse returned by kb_ -> setup_argument(). They are needed by ther kernel_call_traits.
		 */
		std::vector<boost::any> returnee_vec_;
};


inline void kernel::check_number_of_parameters (const int number) {
	if (number_of_parameters_ != number) {
		throw exception::kernel_number_of_parameters_mismatch(number_of_parameters_, number);
	}
}


template <typename P>
void kernel::handle_call_traits(const P &p, const int i) {
	if (dirty[i-1]) {
		typedef typename kernel_device_type<P>::type device_type;
		typedef typename kernel_host_type<P>::type host_type;
		
		device_reference<device_type> device_ref = boost::any_cast< device_reference<device_type> >(returnee_vec_[i-1]);

		// we can are allowed to make this cast
		// because this function is only called, when p is passed by reference to the kernel
		P &temp_p = const_cast<P&>(p);
		kernel_call_traits<host_type, device_type>::dirty(temp_p, device_ref);
	}
}


/***  OPERATPR()  ***/
inline void kernel::operator()(const device &d) {
	UNUSED_PARAMETER(d);
	check_number_of_parameters(0);
	
	kb_ -> configure_call();

	kb_->launch();
}

template< typename P1 >
void kernel::operator()(const device &d, const P1 &p1 ) {
	check_number_of_parameters(1);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );

	kb_->launch();

	handle_call_traits (p1, 1);

	returnee_vec_.clear();
}

template< typename P1, typename P2 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2 ) {
	check_number_of_parameters(2);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);

	returnee_vec_.clear();
}

template< typename P1, typename P2, typename P3 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3 ) {
	check_number_of_parameters(3);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4 ) {
	check_number_of_parameters(4);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5 ) {
	check_number_of_parameters(5);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6 ) {
	check_number_of_parameters(6);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7 ) {
	check_number_of_parameters(7);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8 ) {
	check_number_of_parameters(8);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p8), 8 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);
	handle_call_traits (p8, 8);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9 ) {
	check_number_of_parameters(9);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p8), 8 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p9), 9 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);
	handle_call_traits (p8, 8);
	handle_call_traits (p9, 9);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10 ) {
	check_number_of_parameters(10);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p8), 8 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p9), 9 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p10), 10 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);
	handle_call_traits (p8, 8);
	handle_call_traits (p9, 9);
	handle_call_traits (p10, 10);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11 ) {
	check_number_of_parameters(11);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p8), 8 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p9), 9 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p10), 10 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p11), 11 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);
	handle_call_traits (p8, 8);
	handle_call_traits (p9, 9);
	handle_call_traits (p10, 10);
	handle_call_traits (p11, 11);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12 ) {
	check_number_of_parameters(12);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p8), 8 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p9), 9 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p10), 10 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p11), 11 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p12), 12 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);
	handle_call_traits (p8, 8);
	handle_call_traits (p9, 9);
	handle_call_traits (p10, 10);
	handle_call_traits (p11, 11);
	handle_call_traits (p12, 12);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12, typename P13 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12, const P13 &p13 ) {
	check_number_of_parameters(13);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p8), 8 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p9), 9 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p10), 10 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p11), 11 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p12), 12 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p13), 13 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);
	handle_call_traits (p8, 8);
	handle_call_traits (p9, 9);
	handle_call_traits (p10, 10);
	handle_call_traits (p11, 11);
	handle_call_traits (p12, 12);
	handle_call_traits (p13, 13);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12, typename P13, typename P14 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12, const P13 &p13, const P14 &p14 ) {
	check_number_of_parameters(14);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p8), 8 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p9), 9 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p10), 10 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p11), 11 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p12), 12 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p13), 13 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p14), 14 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);
	handle_call_traits (p8, 8);
	handle_call_traits (p9, 9);
	handle_call_traits (p10, 10);
	handle_call_traits (p11, 11);
	handle_call_traits (p12, 12);
	handle_call_traits (p13, 13);
	handle_call_traits (p14, 14);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12, typename P13, typename P14, typename P15 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12, const P13 &p13, const P14 &p14, const P15 &p15 ) {
	check_number_of_parameters(15);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p8), 8 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p9), 9 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p10), 10 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p11), 11 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p12), 12 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p13), 13 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p14), 14 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p15), 15 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);
	handle_call_traits (p8, 8);
	handle_call_traits (p9, 9);
	handle_call_traits (p10, 10);
	handle_call_traits (p11, 11);
	handle_call_traits (p12, 12);
	handle_call_traits (p13, 13);
	handle_call_traits (p14, 14);
	handle_call_traits (p15, 15);

	returnee_vec_.clear();
}


template< typename P1, typename P2, typename P3, typename P4, typename P5, typename P6, typename P7, typename P8, typename P9, typename P10, typename P11, typename P12, typename P13, typename P14, typename P15, typename P16 >
void kernel::operator()(const device &d, const P1 &p1, const P2 &p2, const P3 &p3, const P4 &p4, const P5 &p5, const P6 &p6, const P7 &p7, const P8 &p8, const P9 &p9, const P10 &p10, const P11 &p11, const P12 &p12, const P13 &p13, const P14 &p14, const P15 &p15, const P16 &p16 ) {
	check_number_of_parameters(16);
	
	kb_ -> configure_call();

	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p1), 1 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p2), 2 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p3), 3 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p4), 4 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p5), 5 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p6), 6 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p7), 7 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p8), 8 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p9), 9 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p10), 10 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p11), 11 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p12), 12 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p13), 13 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p14), 14 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p15), 15 ) );
	returnee_vec_.push_back ( kb_-> setup_argument(d, boost::any(&p16), 16 ) );

	kb_->launch();

	handle_call_traits (p1, 1);
	handle_call_traits (p2, 2);
	handle_call_traits (p3, 3);
	handle_call_traits (p4, 4);
	handle_call_traits (p5, 5);
	handle_call_traits (p6, 6);
	handle_call_traits (p7, 7);
	handle_call_traits (p8, 8);
	handle_call_traits (p9, 9);
	handle_call_traits (p10, 10);
	handle_call_traits (p11, 11);
	handle_call_traits (p12, 12);
	handle_call_traits (p13, 13);
	handle_call_traits (p14, 14);
	handle_call_traits (p15, 15);
	handle_call_traits (p16, 16);

	returnee_vec_.clear();
}

}

#endif

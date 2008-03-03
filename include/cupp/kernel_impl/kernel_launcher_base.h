/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_kernel_launcher_base_H
#define CUPP_KERNEL_IMPL_kernel_launcher_base_H

#include <boost/any.hpp>

// cuda vector types
#include <vector_types.h>

namespace cupp {
namespace kernel_impl {

/**
 * @class kernel_launcher_base
 * @author Björn Knafla: Initial design
 * @author Jens Breitbart
 * @version 0.3
 * @date 03.08.2007
 * @brief A kernel base base class. This is use in cupp::kernel to hide the template parameter needed by @c kernel_launcher_impl.
 */

class kernel_launcher_base {
	public:
		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual void configure_call() = 0;
		
		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual boost::any setup_argument(const device&, const boost::any&, const int ) = 0;

		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual void launch() = 0;

		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual void set_grid_dim   ( const dim3&   ) = 0;

		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual dim3 grid_dim   ( ) = 0;
		
		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual void set_block_dim  ( const dim3&   ) = 0;

		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual dim3 block_dim  ( ) = 0;
		
		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual void set_shared_mem ( const size_t& ) = 0;

		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual size_t shared_mem ( ) = 0;
		
		/**
		 * Virtual destructor
		 */
		virtual ~kernel_launcher_base() {};
};


}
}

#endif //CUPP_KERNEL_IMPL_is_second_level_const_H

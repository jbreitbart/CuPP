/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_kernel_launcher_base_H
#define CUPP_KERNEL_IMPL_kernel_launcher_base_H

#include <boost/any.hpp>
#include <vector>

namespace cupp {
namespace kernel_impl {

/**
 * @class kernel_launcher_base
 * @author Björn Knafla: Initial design and some enlightening comments.
 * @author Jens Breitbart
 * @version 0.2
 * @date 21.07.2007
 * @brief A kernel base base class. This is use in cupp::kernel to hide the template parameter needed by @c kernel_launcher_impl.
 */

class kernel_launcher_base {
	public:
		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual void configure_launch() = 0;
		
		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual boost::any setup_argument( const boost::any &arg, const int pos ) = 0;

		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual void launch() = 0;

		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual std::vector<bool> dirty_parameters() const = 0;
		
		virtual ~kernel_launcher_base() {};
};


}
}

#endif //CUPP_KERNEL_IMPL_is_second_level_const_H

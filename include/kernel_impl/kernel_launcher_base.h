/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_kernel_launcher_base_H
#define CUPP_KERNEL_IMPL_kernel_launcher_base_H

#include <boost/any.hpp>
#include <deque>

namespace cupp {
namespace kernel_impl {

/**
 * @class kernel_launcher_base
 * @author Björn Knafla: Initial design
 * @author Jens Breitbart
 * @version 0.1
 * @date 19.07.2007
 * @brief A kernel base base class. Has some simple functionality, most of it implemented in kernel_launcher_impl
 */

class kernel_launcher_base {
	public:
		virtual void setup_argument( const boost::any &arg, const int pos ) = 0;

		virtual void launch() = 0;

		/**
		 * See in @c kernel_launcher_impl.
		 */
		virtual std::deque<bool> dirty_parameters() const = 0;
		virtual int number_of_param() const = 0;

		virtual ~kernel_launcher_base() {};


	private:
};

/*
template < typename T >
void kernel_launcher_base::setup_agrument(const T &arg) {
	cudaSetupArgument( arg, offset_ );
	offset_ += sizeof( T );
}

template <typename T>
void kernel_launcher_base::put_argument_on_stack(const T &arg) {
	if (stack_used_+sizeof(T) > 256) {
		throw exception::stack_overflow();
	}
	if (cudaSetupArgument(&arg, sizeof(T), stack_used_) != cudaSuccess) {
		throw exception::cuda_runtime_error(cudaGetLastError());
	}
	stack_used_+=sizeof(T);
}*/

}
}

#endif //CUPP_KERNEL_IMPL_is_second_level_const_H

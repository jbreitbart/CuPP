/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_device_reference_H
#define CUPP_device_reference_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif

// CUPP
#include "cupp/common.h"
#include "cupp/runtime.h"
#include "cupp/shared_device_pointer.h"

namespace cupp {

class device;

/**
 * @class device_reference
 * @author Jens Breitbart
 * @version 0.1
 * @date 23.08.2007
 * @platform Host only
 * @brief This is passed to the kernel if a device parameter is passed by reference
 */

template< typename T >
class device_reference {
	public:
		/**
		 * Creates a device reference on the device @a dev reflecting to value @a value.
		 */
		device_reference (const device &dev, const T &value) : dev_(dev), device_value_ptr_ (cupp::malloc<T>()) {
			cupp::copy_host_to_device (device_value_ptr_, &value);
		}

		/**
		 * @return the value to which this references points to.
		 */
		T get() const {
			T returnee;
			cupp::copy_device_to_host (&returnee, device_value_ptr_);
			return returnee;
		}

		/**
		 * @return the device, this reference is valid on
		 */
		const device& get_device() const {
			return dev_;
		}

		/**
		 * @return a pointer to the data on the device
		 */
		shared_device_pointer < T > get_device_ptr() const {
			return device_value_ptr_;
		}

	private:
		/**
		 * The device we live on
		 */
		const device &dev_;
		
		/**
		 * Well ... the memory for our value on the device
		 */
		shared_device_pointer < T > device_value_ptr_;

}; // class device_reference

} // namespace cupp

#endif

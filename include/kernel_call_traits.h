/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_call_traits_H
#define CUPP_kernel_call_traits_H

#include "cupp_runtime.h"
#include "shared_device_pointer.h"

namespace cupp {

/**
 * @class kernel_call_traits
 * @author Jens Breitbart
 * @version 0.2
 * @date 24.07.2007
 * @brief These traits define the behavior of what happens when a kernel is called.
 */

class device;

template <typename host_type, typename device_type>
class kernel_call_traits {
	public:
		/**
		 * Creates a copy of our data for the device in host memory.
		 * @param d The device the kernel will be executed on
		 * @param that The object that is about to be passed to the kernel
		 * @note This function is called when you pass a parameter by value to a kernel.
		 */
		static device_type get_host_based_device_copy (const device &d, const host_type& that) {
			return that.get_host_based_device_copy(d);
		}
		
		/**
		 * Creates a copy of our data for the device in host memory.
		 * @param d The device the kernel will be executed on
		 * @param that The object that is about to be passed to the kernel
		 * @note This function is called when you pass a parameter by reference to a kernel.
		 */
		static shared_device_pointer<device_type> get_device_based_device_copy (const device &d, const host_type& that) {
			return that.get_device_based_device_copy(d);
		}
		
		/**
		 * This function is when the value may have been changed on the device.
		 * @param d The device the kernel will be executed on
		 * @param that The host representation of your data
		 * @param device_copy The pointer you created with @a get_device_based_device_copy
		 * @note This function is only called if you pass a parameter by non-const reference to a kernel.
		 */
		static void dirty (const device &d, const host_type& that, shared_device_pointer<device_type> device_copy) {
			that.dirty(d, device_copy);
		}
};

/**
 * The default traits for all types that need no special handling (device_type == host_type)
 */
template <typename type>
class kernel_call_traits <type, type> {
	public:
		/**
		 * @see above
		 */
		inline static const type& get_host_based_device_copy (const device &d, const type& that) {
			UNUSED_PARAMETER(d);
			return that;
		}
		
		/**
		 * @see above
		 */
		inline static shared_device_pointer<type> get_device_based_device_copy (const device &d, const type& that) {
			// copy device_copy into global memory
			shared_device_pointer<type> device_copy_ptr ( cupp::malloc<type>() );

			/// @todo is this legal?
			cupp::copy_host_to_device(device_copy_ptr, &get_host_based_device_copy(d, that));

			return device_copy_ptr;
		}

		/**
		* @see above
		*/
		inline static void dirty (const device &d, const type& that, shared_device_pointer<type> device_copy) {
			UNUSED_PARAMETER(d);
			// do a dirty ugly bit copy from device memory to host memory
			cupp::copy_device_to_host ( const_cast<type*>(&that), device_copy );
		}
};


} // cupp

#endif //CUPP_kernel_call_traits_H

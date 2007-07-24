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

template <typename host_type, typename device_type>
class kernel_call_traits {
	public:
		/**
		 * Creates a copy of our data for the device in host memory.
		 * @note This function is called when you pass a parameter by value to a kernel.
		 */
		static device_type get_host_based_device_copy (const host_type& that) {
			return that.get_host_based_device_copy();
		}
		
		/**
		 * Creates a copy of our data for the device in host memory.
		 * @note This function is called when you pass a parameter by reference to a kernel.
		 */
		static shared_device_pointer<device_type> get_device_based_device_copy (const host_type& that) {
			return that.get_device_based_device_copy();
		}
		
		/**
		* This function is when the value may have been changed on the device.
		* @param that The host representation of your data
		* @param device_copy The pointer you created with @a get_device_based_device_copy
		* @note This function is only called if you pass a parameter by non-const reference to a kernel.
		*/
		static void dirty (const host_type& that, shared_device_pointer<device_type> device_copy) {
			that.dirty(device_copy);
		}
};

// this is the default trait for all types which require no special treatment
template <typename type>
class kernel_call_traits <type, type> {
	public:
		/**
		 * @see above
		 */
		inline static const type& get_host_based_device_copy (const type& that) {
			return that;
		}
		
		/**
		 * @see above
		 */
		inline static shared_device_pointer<type> get_device_based_device_copy (const type& that) {
			// copy device_copy into global memory
			shared_device_pointer<type> device_copy_ptr ( cupp::malloc<type>() );

			/// @todo is this legal?
			cupp::copy_host_to_device(device_copy_ptr, &get_host_based_device_copy(that));

			return device_copy_ptr;
		}

		/**
		* @see above
		*/
		inline static void dirty (const type& that, shared_device_pointer<type> device_copy) {
			// do a dirty ugly bit copy from device memory to host memory
			cupp::copy_device_to_host ( const_cast<type*>(&that), device_copy );
		}
};


} // cupp

#endif //CUPP_kernel_call_traits_H

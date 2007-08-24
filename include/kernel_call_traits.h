/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_call_traits_H
#define CUPP_kernel_call_traits_H

#include "cupp_runtime.h"
#include "device.h"
#include "device_reference.h"

namespace cupp {

/**
 * @class kernel_call_traits
 * @author Jens Breitbart
 * @version 0.3.1
 * @date 23.08.2007
 * @brief These traits define the behavior of what happens when a kernel is called.
 */

template <typename host_type, typename device_type>
struct kernel_call_traits {

	/**
	 * Transforms the host type to the device type
	 * @param d The device the kernel will be executed on
	 * @param that The object that is about to be passed to the kernel
	 * @note This function is called when you pass a parameter by value to a kernel.
	 */
	static device_type transform (const device &d, const host_type& that) {
		return that.transform (d);
	}

	/**
	 * Transforms the device type to the host type
	 * @param d The device the kernel was executed on
	 * @param that The object that was passed to the kernel
	 * @note This function is called when you pass a parameter by value to a kernel and the kernel has been started.
	 */
	static device_type transform (const device &d, const device_type& that) {
		return that.transform (d);
	}
	
	/**
	 * Creates a device reference to be passed to the kernel
	 * @param d The device the kernel will be executed on
	 * @param that The object that is about to be passed to the kernel
	 * @note This function is called when you pass a parameter by reference to a kernel.
	 */
	static device_reference<device_type> get_device_reference (const device &d, const host_type& that) {
		return that.get_device_reference (d);
	}

	/**
	 * This function is called when the value may have been changed on the device.
	 * @param d The device the kernel will be executed on
	 * @param that The host representation of your data
	 * @param device_copy The pointer you created with @a get_device_based_device_copy
	 * @note This function is only called if you pass a parameter by non-const reference to a kernel.
	 */
	static void dirty (const host_type& that, device_reference<device_type> device_ref) {
		that.dirty(device_ref);
	}
};

/**
 * The default traits for all types that need no special handling (device_type == host_type)
 */
template <typename type>
struct kernel_call_traits <type, type> {

	/**
	 * @see above
	 */
	static const type& transform (const device &d, const type& that) {
		/// @todo is there a reason why we should not return by ref?
		UNUSED_PARAMETER(d);
		return that;
	}

	/**
	 * @see above
	 */
	static device_reference<type> get_device_reference (const device &d, const type& that) {
		return device_reference<type> (d, that);
	}

	/**
	 * @see above
	 */
	static void dirty (const type& that, device_reference<type> device_ref) {
		type& temp = const_cast<type&>(that);
		temp = device_ref.get();
	}
};


} // cupp

#endif //CUPP_kernel_call_traits_H

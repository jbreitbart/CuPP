/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_call_traits_H
#define CUPP_kernel_call_traits_H


namespace cupp {

/**
 * @class kernel_call_traits
 * @author Jens Breitbart
 * @version 0.1
 * @date 22.07.2007
 * @brief These traits define the behavior of what happens when a kernel is called.
 */

template <typename host_type, typename device_type>
class kernel_call_traits {
	public:
		/**
		* This function is called when a parameter of type @a host_type is passed as a not-const reference
		* to a kernel.
		* @param that the host representation of our data
		* @param device_copy a pointer to the dirty data on the device (this is a DEVICE POINTER, treat it with care!)
		*/
		inline static void dirty (host_type& that, const device_type *device_copy);

		/**
		* Creates a copy of our data for the device
		*/
		inline static const device_type get_device_copy (const host_type& that);

};

// this is the default trait for all types which require no special treatment
template <typename type>
class kernel_call_traits <type, type> {
	public:
		/**
		* This function is called when a parameter of type @a host_type is passed as a not-const reference
		* to a kernel.
		* @param that the host representation of our data
		* @param device_copy a pointer to the dirty data on the device (this is a DEVICE POINTER, treat it with care!)
		*/
		inline static void dirty (type& that, const type *device_copy) {
			// do a dirty ugly bit copy from device memory to host memory
			if (cudaMemcpy (&that, device_copy, sizeof(type), cudaMemcpyDeviceToHost) != cudaSuccess) {
				throw exception::cuda_runtime_error(cudaGetLastError());
			}
			
			if (cudaFree(device_copy) != cudaSuccess) {
				throw exception::cuda_runtime_error(cudaGetLastError());
			}
		}

		/**
		* Creates a copy of our data for the device
		*/
		inline static const type& get_device_copy (const type& that) {
			
			return that;
		}

};


} // cupp

#endif //CUPP_kernel_call_traits_H
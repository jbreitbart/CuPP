/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_DEVICET_memory1d_H
#define CUPP_DEVICET_memory1d_H

// Include std::size_t
#include <stddef.h>

#include "cupp/common.h"

namespace cupp {

template <typename T>
class memory1d;

namespace deviceT {

/**
 * @class memory1d
 * @author Jens Breitbart
 * @version 0.1
 * @date 10.07.2007
 * @brief Represents a memory block on an associated CUDA device.
 * @platform Device only
 */

template< typename T, typename host_type_=cupp::memory1d<T> >
class memory1d {
	public:
		/**
		 * Set up the type bindings
		 */
		typedef memory1d<T>   device_type;
		typedef host_type_    host_type;

		/**
		 * @typedef size_type
		 * @brief The type you should use to index this class
		 */
		typedef std::size_t size_type;

		/**
		 * @typedef value_type
		 * @brief The type of data you want to store
		 */
		typedef T value_type;


		/**
		 * @brief Returns the size of the memory block
		 * @platform Host
		 * @platform Device
		 */
		CUPP_RUN_ON_HOST CUPP_RUN_ON_DEVICE
		size_type size() const;


		/**
		 * @brief Access the memory
		 * @param index The index of the element you want to access
		 * @platform Device
		 */
		CUPP_RUN_ON_DEVICE
		T& operator[]( const size_type size_type );

		/**
		 * @brief Access the memory
		 * @param index The index of the element you want to access
		 * @platform Device
		 */
		CUPP_RUN_ON_DEVICE
		T const& operator[]( const size_type index ) const;

		CUPP_RUN_ON_HOST
		void set_device_pointer( T* device_pointer );

		CUPP_RUN_ON_HOST
		void set_size(const size_type size);

	/*private:*/
		/**
		 * The pointer to the device memory
		 */
		T* device_pointer_;

		/**
		 * How many memory has been allocated
		 */
		size_type size_;

}; // class memory1d


template <typename T, typename host_type>
T& memory1d<T, host_type>::operator[](const size_type index) {
	return device_pointer_[index];
}

template <typename T, typename host_type>
T const& memory1d<T, host_type>::operator[](const size_type index) const {
	return device_pointer_[index];
}


template <typename T, typename host_type>
typename memory1d<T, host_type>::size_type memory1d<T, host_type>::size() const {
	return size_;
}

template <typename T, typename host_type>
void memory1d<T, host_type>::set_device_pointer(T* device_pointer) {
	device_pointer_ = device_pointer;
}

template <typename T, typename host_type>
void memory1d<T, host_type>::set_size(const size_type size) {
	size_ = size;
}

} // namespace deviceT
} // namespace cupp

#endif

/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_DEVICET_memory1d_H
#define CUPP_DEVICET_memory1d_H

// Include std::size_t
#include <cstddef>

// Include std::swap
#include <algorithm>


#include "cupp_common.h"

namespace cupp {
namespace deviceT {

/**
 * @class memory1d
 * @author Jens Breitbart
 * @version 0.1
 * @date 10.07.2007
 * @brief Represents a changeable memory block on an associated CUDA device.
 * @platform Device only
 *
 * askk
 */

template< typename T >
class memory1d {
	public:
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
		CUPP_HOST CUPP_DEVICE
		size_type size() const;


		/**
		 * @brief Access the memory
		 * @param index The index of the element you want to access
		 * @platform Device
		 * @todo How to implement this for the host? 
		 */
		CUPP_DEVICE
		T& operator[]( size_type size_type );

		/**
		 * @brief Access the memory
		 * @param index The index of the element you want to access
		 * @warning @a out_iter must be able to hold at least @c size() elements.
		 * @platform Device
		 */
		CUPP_DEVICE
		T const& operator[]( size_type index ) const;

	private:
		/**
		 * The pointer to the device memory
		 */
		T* device_pointer_;

		/**
		 * How many memory has been allocated
		 */
		size_type size_;
}; // class memory1d


template <typename T>
T& memory1d<T>::operator[](size_type index) {
	return device_pointer_[index];
}

template <typename T>
T const& memory1d<T>::operator[](size_type index) const {
	return device_pointer_[index];
}


template <typename T>
typename memory1d<T>::size_type memory1d<T>::size() const {
	return size_;
}


} // namespace deviceT
} // namespace cupp

#endif

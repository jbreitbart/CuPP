/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_DEVICET_vector_H
#define CUPP_DEVICET_vector_H

// Include std::size_t
#include <cstddef>

#include "cupp_common.h"

namespace cupp {

template <typename T>
class vector;

namespace deviceT {

/**
 * @class vector
 * @author Jens Breitbart
 * @version 0.1
 * @date 24.07.2007
 * @platform Device only
 */

template< typename T >
class vector  {
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
		 * Creates an empty and useless vector :-)
		 */
		vector () {}
		
		/**
		 * Constructor
		 * @param size The size of the memory to be pointed to
		 * @param device_pointer The pointer to the memory (device pointer!)
		 */
		vector ( size_type size, T* device_pointer) : size_(size), device_pointer_(device_pointer) {}
		
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
		 * @platform Device only
		 * @todo How to implement this for the host? 
		 */
		CUPP_RUN_ON_DEVICE
		T& operator[]( size_type size_type );

		/**
		 * @brief Access the memory
		 * @param index The index of the element you want to access
		 * @warning @a out_iter must be able to hold at least @c size() elements.
		 * @platform Device only
		 */
		CUPP_RUN_ON_DEVICE
		T const& operator[]( size_type index ) const;
		
	private:

		/**
		 * How many memory has been allocated
		 */
		size_type size_;

		/**
		 * The pointer to the device memory
		 */
		T* device_pointer_;
};

template <typename T>
T& vector<T>::operator[](size_type index) {
	return device_pointer_[index];
}

template <typename T>
T const& vector<T>::operator[](size_type index) const {
	return device_pointer_[index];
}


template <typename T>
typename vector<T>::size_type vector<T>::size() const {
	return size_;
}


} // namespace deviceT
} // namespace cupp

#endif

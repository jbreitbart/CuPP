/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_DEVICET_vector_H
#define CUPP_DEVICET_vector_H

#include "cupp/common.h"
#include "cupp/kernel_type_binding.h"
#include "cupp/deviceT/memory1d.h"

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
class vector/* : public memory1d<T, cupp::vector<T> >*/ {
	public:
		/**
		 * Set up the type bindings
		 */
		typedef vector<T>                                          device_type;
		typedef cupp::vector< typename get_type<T>::host_type >    host_type;
		
		/**
		 * @typedef size_type
		 * @brief The type you should use to index this class
		 */
		typedef typename memory1d<T, cupp::vector<T> >::size_type size_type;

		/**
		 * @typedef value_type
		 * @brief The type of data you want to store
		 */
		typedef typename memory1d<T, cupp::vector<T> >::value_type value_type;

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

};

template <typename T>
T& vector<T>::operator[](const size_type index) {
	return device_pointer_[index];
}

template <typename T>
T const& vector<T>::operator[](const size_type index) const {
	return device_pointer_[index];
}


template <typename T>
typename vector<T>::size_type vector<T>::size() const {
	return size_;
}

template <typename T>
void vector<T>::set_device_pointer(T* device_pointer) {
	device_pointer_ = device_pointer;
}

template <typename T>
void vector<T>::set_size(const size_type size) {
	size_ = size;
}

} // namespace deviceT
} // namespace cupp

#endif

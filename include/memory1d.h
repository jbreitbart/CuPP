/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_memory1d_H
#define CUPP_memory1d_H

// Include std::size_t
#include <cstddef>

// Include std::swap
#include <algorithm>

#include <vector>

#include "cupp_common.h"

#if !defined(__CUDACC__)
	#include "exception/cuda_runtime_error.h"
	#include "exception/memory_access_violation.h"
#endif

#include <cuda_runtime.h>


namespace cupp {

// Just used to force the user to configure and get a device.
class device;

/**
 * @class memory1d
 * @author Björn Knafla: Initial design
 * @author Jens Breitbart
 * @version 0.2
 * @date 20.06.2007
 * @brief Represents a memory block on an associated CUDA device.
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

		// dev is a pure dummy, it is only used to force the user to configure a device
		// before creating memory on it.
#if !defined(__CUDACC__)
		/**
		 * @brief Associates memory for @a size elements on the device @a dev
		 * @param dev The device on which you want to allocate memory
		 * @param size How many elements you store on the device
		 * @exception cuda_runtime_error
		 * @platform Host only
		 */
		CUPP_HOST
		memory1d( device const& dev, size_type size );
		
		/**
		 * @brief Associates memory for @a size elements on the device @a dev and fills it with the byte @a init_value
		 * @param dev The device on which you want to allocate memory
		 * @param init_value The initialization value for the memory
		 * @param size How many elements you store on the device
		 * @exception cuda_runtime_error
		 * @platform Host only
		 */
		CUPP_HOST
		memory1d( device const& dev, int init_value, size_type size );
		
		/**
		 * @brief Associates memory for @a size elements on the device @a dev and fills it with the data pointed by @a data
		 * @param dev The device on which you want to allocate memory
		 * @param data The data which will get transfered to the GPU
		 * @param size How many elements you store on the device
		 * @exception cuda_runtime_error
		 * @warning Be sure that @a data points to at least @a size many elements.
		 * @platform Host only
		 */
		CUPP_HOST
		memory1d( device const& dev, T const* data, size_type size );
		
		/**
		 * @brief Associates memory on the device @a dev and copies all elements between @a first and @a last to that memory.
		 * @param dev The device on which you want to allocate memory
		 * @param first The starting point of your data
		 * @param last The end+1 of your data
		 * @exception cuda_runtime_error
		 * @platform Host only
		 */
		template <typename InputIterator>
		memory1d( device const& dev, InputIterator first, InputIterator last );

		/**
		 * @brief Creates a new memory block on the device and copies the data of @a other to the new block.
		 * @param other The memory that will be copied.
		 * @exception cuda_runtime_error
		 * @platform Host only
		 */
		CUPP_HOST
		memory1d( memory1d<T> const& other );

		/**
		 * @brief Frees the memory on the device.
		 * @exception cuda_runtime_error
		 * @platform Host only
		 */
		CUPP_HOST
		~memory1d();
#endif

		/**
		 * @brief Copies the data from @a other to its own memory block.
		 * @param other The data you want to copy
		 * @todo how to handle different sizes?
		 * @platform Host
		 */
		CUPP_HOST
		memory1d< T >& operator=( const memory1d< T > &other );
		

		/**
		 * @brief Swaps the data between @a other and @a this.
		 * @param other The data you want to swap
		 * @platform Host
		 * @platform Device
		 */
		CUPP_HOST CUPP_DEVICE
		void swap( memory1d& other );
		

		/**
		 * @brief Returns the size of the memory block
		 * @platform Host
		 * @platform Device
		 */
		CUPP_HOST CUPP_DEVICE
		size_type size() const;
		

		/**
		 * @brief Set the memory block to the byte value of @a value
		 * @param value The byte value to be set.
		 * @platform Host only
		 */
		CUPP_HOST
		void set( int value );


		/**
		 * @brief Copies data to the memory on the device
		 * @param first The starting point of your data
		 * @param last The end+1 of your data
		 * @param offset Is non-byte offset (TM)
		 * @platform Host only
		 * @todo We could resize the memory if last-first > size
		 */
#if !defined(__CUDACC__)
		template <typename InputIterator>
		void copy_to_device( InputIterator first, InputIterator last, int offset );
#endif
		
		/**
		 * @brief Copies data to the memory on the device
		 * @param data The data which will get transfered to the device
		 * @param offset Is non-byte offset (TM)
		 * @warning Be sure that @a data points to at least @a this.size() many elements.
		 * @platform Host only
		 */
		CUPP_HOST
		void copy_to_device( T const* data, size_type offset=0 );
		
		/**
		 * @brief Copies data to the memory on the device
		 * @param data The data which will get transfered to the device
		 * @param count How many data you want to transfer
		 * @param offset Is non-byte offset (TM)
		 * @warning Be sure that @a data points to at least @a count() many elements.
		 * @platform Host only
		 * @todo We could resize the memory if count+offset > size
		 */
		CUPP_HOST
		void copy_to_device( size_type count, T const* data, size_type offset=0 );

		/**
		 * @brief Copies data to the memory on the device
		 * @param other The memory that will be copied.
		 * @param offset Is non-byte offset (TM)
		 * @platform Host only
		 * @todo We could resize the memory if other.size > size
		 */
		CUPP_HOST
		void copy_to_device( memory1d const& other, size_type offset=0 );
		
		/**
		 * @brief Copies data to the memory on the device
		 * @param other The memory that will be copied.
		 * @param count How many elements will be copied
		 * @param offset Is non-byte offset (TM)
		 * @platform Host only
		 * @todo We could resize the memory if @c other.size() != @c size()
		 */
		CUPP_HOST
		void copy_to_device( memory1d const& other, size_type count, size_type offset=0 );
		

		/**
		 * @brief Copies data from the memory on the device to @a destination
		 * @param destination The place where you want to store the data
		 * @warning Be sure that @a destination points to at least @c size() many elements.
		 * @platform Host only
		 */
		CUPP_HOST
		void copy_to_host( T* destination );
		
		/**
		 * @brief Copies data from the memory on the device to @a out_iter
		 * @param out_iter An output iterator where you want the data to be stored
		 * @warning @a out_iter must be able to hold at least @c size() elements.
		 * @platform Host only
		 */
#if !defined(__CUDACC__)
		template <typename OutputIterator>
		void copy_to_host( OutputIterator out_iter );
#endif

#if 0
		// Be strongly cautioned not to use this!!!!!!
		//Perhaps we should just use the functions below (your idea! I really like it!)
		CUPP_HOST CUPP_DEVICE
		T* cuda_pointer() const {return device_pointer_;}
#endif

/// @code_review we should discuss this :-)
#if defined(__CUDACC__)
		/**
		 * @brief Access the memory
		 * @param index The index of the element you want to access
		 * @platform Device
		 * @todo How to implement this for the host? 
		 */
		CUPP_DEVICE
		T& operator[]( size_type size_type );
#endif

		/**
		 * @brief Access the memory
		 * @param index The index of the element you want to access
		 * @warning @a out_iter must be able to hold at least @c size() elements.
		 * @platform Device
		 * @platform Host
		 */
		CUPP_HOST CUPP_DEVICE
		T const& operator[]( size_type index ) const;
		
	private:
		/**
		 * @brief Allocates memory on the device
		 * @exception cuda_runtime_error
		 */
		CUPP_HOST
		void malloc();

		/**
		 * @brief Free the memory on the device
		 * @exception cuda_runtime_error
		 */
		CUPP_HOST
		void free();

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

#if !defined(__CUDACC__)
template <typename T>
memory1d<T>::memory1d( device const& dev, size_type size ) : device_pointer_(0), size_(size) {
	malloc();
}


template <typename T>
memory1d<T>::memory1d( device const& dev, int init_value, size_type size ) : device_pointer_(0), size_(size) {
	malloc();
	set(init_value);
}


template <typename T>
memory1d<T>::memory1d( device const& dev, T const* data, size_type size ) : device_pointer_(0), size_(size) {
	malloc();
	copy_to_device(data);
}


#if !defined(__CUDACC__)
	template <typename T>
	template <typename InputIterator>
	memory1d<T>::memory1d( device const& dev, InputIterator first, InputIterator last ) : device_pointer_(0), size_(size) {
		malloc();
		copy_to_device(first, last);
	}
#endif

template <typename T>
memory1d<T>::memory1d( memory1d<T> const& other ) : device_pointer_(0), size_(other.size()) {
	malloc();
	copy_to_device(other);
}


template <typename T>
memory1d<T>::~memory1d() {
	free();
}
#endif

template <typename T>
memory1d< T >& memory1d<T>::operator=( const memory1d< T > &other ) {
	copy_to_device(other);
}


template <typename T>
void memory1d<T>::swap( memory1d& other ) {
	std::swap(this.device_pointer_, other.device_pointer_);
	std::swap(this.size_, other.size_);
}


template <typename T>
typename memory1d<T>::size_type memory1d<T>::size() const {
	return size_;
}


template <typename T>
void memory1d<T>::set(int value) {
	if (cudaMemset( reinterpret_cast<void*>( device_pointer_ ), value, sizeof(T)*size() ) != cudaSuccess) {
		#if !defined(__CUDACC__)
			throw exception::cuda_runtime_error(cudaGetLastError());
		#endif
	}
}


template <typename T>
void memory1d<T>::malloc() {
	if (cudaMalloc( reinterpret_cast<void**>( &device_pointer_ ), sizeof(T)*size() ) != cudaSuccess) {
		#if !defined(__CUDACC__)
			throw exception::cuda_runtime_error(cudaGetLastError());
		#endif
	}
}


template <typename T>
void memory1d<T>::free() {
	if (cudaFree(device_pointer_) != cudaSuccess) {
		#if !defined(__CUDACC__)
			throw exception::cuda_runtime_error(cudaGetLastError());
		#endif
	}
}


#if !defined(__CUDACC__)
	template <typename T>
	template <typename InputIterator>
	void memory1d<T>::copy_to_device( InputIterator first, InputIterator last, int offset ) {
		std::vector<T> temp;
		temp.assign(first, last);
		copy_to_device(temp.size(), &temp[0], offset);
	}
#endif


template <typename T>
void memory1d<T>::copy_to_device( T const* data, size_type offset ) {
	copy_to_device(size(), data, offset);
}


template <typename T>
void memory1d<T>::copy_to_device( size_type count, T const* data, size_type offset) {
	if (count + offset > size()) {
		#if !defined(__CUDACC__)
			throw exception::memory_access_violation();
		#endif
	}
	
	if ( cudaMemcpy(device_pointer_+offset, data, count * sizeof(T), cudaMemcpyHostToDevice) != cudaSuccess) {
		#if !defined(__CUDACC__)
			throw exception::cuda_runtime_error(cudaGetLastError());
		#endif
	}
}

template <typename T>
void memory1d<T>::copy_to_device (memory1d const& other, size_type offset) {
	copy_to_device(other, other.size(), offset);
}


template <typename T>
void memory1d<T>::copy_to_device (memory1d const& other, size_type count, size_type offset) {
	if (count+offset > size()) {
		#if !defined(__CUDACC__)
			throw exception::memory_access_violation();
		#endif
	}

	if ( cudaMemcpy(device_pointer_+offset, other.device_pointer_, count * sizeof(T), cudaMemcpyDeviceToDevice) != cudaSuccess) {
		#if !defined(__CUDACC__)
			throw exception::cuda_runtime_error(cudaGetLastError());
		#endif
	}
}


template <typename T>
void memory1d<T>::copy_to_host (T* destination) {
	if (cudaMemcpy(destination, device_pointer_, size() * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) {
		#if !defined(__CUDACC__)
			throw exception::cuda_runtime_error(cudaGetLastError());
		#endif
	}
}

#if !defined(__CUDACC__)
	template <typename T>
	template <typename OutputIterator>
	void memory1d<T>::copy_to_host(OutputIterator out_iter) {
		std::vector<T> temp( size() );

		copy_to_host(&temp[0]);

		std::copy(temp.begin(), temp.end(), out_iter);
	}
#endif


#if defined(__CUDACC__)
	template <typename T>
	T& memory1d<T>::operator[](size_type index) {
		return device_pointer_[index];
	}
#endif

template <typename T>
T const& memory1d<T>::operator[](size_type index) const {
	#if defined(__CUDACC__)
		return device_pointer_[index];
	#endif
	#if !defined(__CUDACC__)
		T returnee;
		copy_to_device(1, &returnee, index);
		return returnee;
	#endif
}

} // namespace cupp

#endif

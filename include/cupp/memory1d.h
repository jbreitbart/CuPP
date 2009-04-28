/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_memory1d_H
#define CUPP_memory1d_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif

// CUPP
#include "cupp/common.h"
#include "cupp/runtime.h"
#include "cupp/kernel_type_binding.h"
#include "cupp/shared_device_pointer.h"
#include "cupp/device_reference.h"

#include "cupp/deviceT/memory1d.h"

#include "cupp/exception/cuda_runtime_error.h"
#include "cupp/exception/memory_access_violation.h"

// STD
#include <cstddef> // Include std::size_t
#include <algorithm> // Include std::swap
#include <vector>


namespace cupp {


// Just used to force the user to configure and get a device.
class device;


/**
 * @class memory1d
 * @author Björn Knafla: Initial design
 * @author Jens Breitbart
 * @version 0.2
 * @date 20.06.2007
 * @platform Host only
 * @brief Represents a memory block on an associated CUDA device.
 */

template< typename T >
class memory1d {
	public:
		/**
		 * Set up the type bindings
		 */
		typedef deviceT::memory1d<typename get_type<T>::device_type >  device_type;
		typedef memory1d<T>                                            host_type;
		
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
		/**
		 * @brief Associates memory for @a size elements on the device @a dev
		 * @param dev The device on which you want to allocate memory
		 * @param size How many elements you store on the device
		 * @exception cuda_runtime_error
		 * @platform Host only
		 */
		memory1d( device const& dev, size_type size );
		
		/**
		 * @brief Associates memory for @a size elements on the device @a dev and fills it with the byte @a init_value
		 * @param dev The device on which you want to allocate memory
		 * @param init_value The initialization value for the memory
		 * @param size How many elements you store on the device
		 * @exception cuda_runtime_error
		 * @platform Host only
		 */
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
		memory1d( memory1d<T> const& other );

		/**
		 * @brief Frees the memory on the device.
		 * @exception cuda_runtime_error
		 * @platform Host only
		 */
		~memory1d();

		/**
		 * @brief Copies the data from @a other to its own memory block.
		 * @param other The data you want to copy
		 * @todo how to handle different sizes?
		 * @platform Host
		 */
		memory1d< T >& operator=( const memory1d< T > &other );

		/**
		 * @brief Swaps the data between @a other and @a this.
		 * @param other The data you want to swap
		 * @platform Host
		 * @platform Device
		 */
		void swap( memory1d& other );
		

		/**
		 * @brief Returns the size of the memory block
		 * @platform Host
		 * @platform Device
		 */
		size_type size() const;
		

		/**
		 * @brief Set the memory block to the byte value of @a value
		 * @param value The byte value to be set.
		 * @platform Host only
		 */
		void set( int value );


		/**
		 * @brief Copies data to the memory on the device
		 * @param first The starting point of your data
		 * @param last The end+1 of your data
		 * @param offset Is non-byte offset (TM)
		 * @platform Host only
		 * @todo We could resize the memory if last-first > size
		 */
		template <typename InputIterator>
		void copy_to_device( InputIterator first, InputIterator last, int offset );
		
		/**
		 * @brief Copies data to the memory on the device
		 * @param data The data which will get transfered to the device
		 * @param offset Is non-byte offset (TM)
		 * @warning Be sure that @a data points to at least @a this.size() many elements.
		 * @platform Host only
		 */
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
		void copy_to_device( size_type count, T const* data, size_type offset=0 );

		/**
		 * @brief Copies data to the memory on the device
		 * @param other The memory that will be copied.
		 * @param offset Is non-byte offset (TM)
		 * @platform Host only
		 * @todo We could resize the memory if other.size > size
		 */
		void copy_to_device( memory1d const& other, size_type offset=0 );
		
		/**
		 * @brief Copies data to the memory on the device
		 * @param other The memory that will be copied.
		 * @param count How many elements will be copied
		 * @param offset Is non-byte offset (TM)
		 * @platform Host only
		 * @todo We could resize the memory if @c other.size() != @c size()
		 */
		void copy_to_device( memory1d const& other, size_type count, size_type offset=0 );
		

		/**
		 * @brief Copies data from the memory on the device to @a destination
		 * @param destination The place where you want to store the data
		 * @warning Be sure that @a destination points to at least @c size() many elements.
		 * @platform Host only
		 */
		void copy_to_host( T* destination );
		
		/**
		 * @brief Copies data from the memory on the device to @a out_iter
		 * @param out_iter An output iterator where you want the data to be stored
		 * @warning @a out_iter must be able to hold at least @c size() elements.
		 * @platform Host only
		 */
		template <typename OutputIterator>
		void copy_to_host( OutputIterator out_iter );

		/**
		 * @return A shared device pointer to the memory handled by @a this
		 */
		shared_device_pointer<T> cuda_pointer() const {  return device_pointer_;  }

		/**
		 * @return the device the memory is allocated on
		 */
		const device& get_device() const { return d_; }


	public: /*** CuPP kernel call traits implementation ***/
		/**
		 * @brief This function is called by the kernel_call_traits
		 * @return A on the device useable memory1d reference
		 */
		device_type transform(const device &d);

		/**
		 * @brief This function is called by the kernel_call_traits
		 * @return A on the device useable memory1d reference
		 */
		device_reference<device_type> get_device_reference(const device &d);
		
		/**
		 * @brief This function is called by the kernel_call_traits
		 */
		void dirty (device_reference<device_type> device_ref) {
			UNUSED_PARAMETER(device_ref);
		}


	private:
		/**
		 * The pointer to the device memory
		 */
		shared_device_pointer<T> device_pointer_;

		/**
		 * How much memory has been allocated
		 */
		size_type size_;

		/**
		 * Our proxy on the device
		 */
		mutable device_reference< device_type > *device_ref_;

		/**
		 * The device we live on
		 */
		const device& d_;

}; // class memory1d


template <typename T>
typename memory1d<T>::device_type memory1d<T>::transform(const device &d) {
	UNUSED_PARAMETER(d);
	
	device_type temp;

	temp.set_size (size());
	temp.set_device_pointer (cuda_pointer().get());
	
	return temp;
}

template <typename T>
device_reference< typename memory1d<T>::device_type > memory1d<T>::get_device_reference(const device &d) {
	if (device_ref_ == 0) {
		// copy device_copy into global memory
		device_ref_ = new device_reference < device_type > (d, transform(d) );
	}

	return *device_ref_;
}


template <typename T>
memory1d<T>::memory1d( device const& dev, size_type size ) : device_pointer_( cupp::malloc<T>(size) ), size_(size), device_ref_(0), d_(dev) {}


template <typename T>
memory1d<T>::memory1d( device const& dev, int init_value, size_type size ) : device_pointer_( cupp::malloc<T>(size) ), size_(size), device_ref_(0), d_(dev) {
	set(init_value);
}


template <typename T>
memory1d<T>::memory1d( device const& dev, T const* data, size_type size ) : device_pointer_( cupp::malloc<T>(size) ), size_(size), device_ref_(0), d_(dev) {
	UNUSED_PARAMETER(dev);
	copy_to_device(data);
}


template <typename T>
template <typename InputIterator>
memory1d<T>::memory1d( device const& dev, InputIterator first, InputIterator last ): device_ref_(0), d_(dev) {
	size_ = last - first;
	
	device_pointer_( cupp::malloc<T>(size_) );
	copy_to_device(first, last);
}

template <typename T>
memory1d<T>::memory1d( memory1d<T> const& other ) : device_pointer_( cupp::malloc<T>(other.size()) ), size_(other.size()), device_ref_(0), d_(other.get_device()) {
	copy_to_device(other);
}


template <typename T>
memory1d<T>::~memory1d() {
	delete device_ref_;
}

template <typename T>
memory1d< T >& memory1d<T>::operator=( const memory1d< T > &other ) {
	copy_to_device(other);
}


template <typename T>
void memory1d<T>::set(int value) {
	cupp::mem_set (device_pointer_, value, size());
}


template <typename T>
template <typename InputIterator>
void memory1d<T>::copy_to_device( InputIterator first, InputIterator last, int offset ) {
	std::vector<T> temp;
	temp.assign(first, last);
	copy_to_device(temp.size(), &temp[0], offset);
}


template <typename T>
void memory1d<T>::copy_to_device( T const* data, size_type offset ) {
	copy_to_device(size(), data, offset);
}


template <typename T>
void memory1d<T>::copy_to_device( size_type count, T const* data, size_type offset) {
	if (count + offset > size()) {
		throw exception::memory_access_violation();
	}

	cupp::copy_host_to_device (device_pointer_.get()+offset, data, count);
}

template <typename T>
void memory1d<T>::copy_to_device (memory1d const& other, size_type offset) {
	copy_to_device(other, other.size(), offset);
}


template <typename T>
void memory1d<T>::copy_to_device (memory1d const& other, size_type count, size_type offset) {
	if (count+offset > size()) {
		throw exception::memory_access_violation();
	}

	cupp::copy_device_to_device (device_pointer_.get()+offset, other.device_pointer_, count);
}


template <typename T>
void memory1d<T>::copy_to_host (T* destination) {
	cupp::copy_device_to_host (destination, device_pointer_, size() );
}


template <typename T>
template <typename OutputIterator>
void memory1d<T>::copy_to_host(OutputIterator out_iter) {
	std::vector<T> temp( size() );

	copy_to_host(&temp[0]);

	std::copy(temp.begin(), temp.end(), out_iter);
}



template <typename T>
void memory1d<T>::swap( memory1d& other ) {
	swap(this.device_pointer_, other.device_pointer_);
	std::swap(this.size_, other.size_);
}


template <typename T>
typename memory1d<T>::size_type memory1d<T>::size() const {
	return size_;
}


} // namespace cupp

#endif

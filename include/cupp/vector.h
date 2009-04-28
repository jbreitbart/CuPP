/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_vector_H
#define CUPP_vector_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif

// CUPP
#include "cupp/common.h"
#include "cupp/runtime.h"
#include "cupp/kernel_type_binding.h"
#include "cupp/kernel_call_traits.h"
#include "cupp/device.h"
#include "cupp/memory1d.h"

#include "cupp/deviceT/vector.h"

#include "cupp/exception/cuda_runtime_error.h"
#include "cupp/exception/memory_access_violation.h"

// STD
#include <cstddef> // Include std::size_t
#include <algorithm> // Include std::swap
#include <vector>

// CUDA
#include <cuda_runtime.h>

namespace cupp {

// Just used to force the user to configure and get a device.
class device;


/**
 * @class vector
 * @author Jens Breitbart
 * @version 0.2
 * @date 24.08.2007
 * @platform Host only
 * @brief A std::vector wrapper, which can be transfered to the device incl. lazy memory copying.
 */

template< typename T >
class vector {
	private:
		typedef typename get_type<T>::device_type                T_device_type;
		
	public: /*** TYPEDEFS  ***/
		typedef deviceT::vector< T_device_type >                 device_type;
		typedef vector<T>                                        host_type;

		typedef typename std::vector<T>::size_type               size_type;
		typedef typename std::vector<T>::value_type              value_type;
		typedef typename std::vector<T>::const_iterator          const_iterator;
		typedef typename std::vector<T>::const_reverse_iterator  const_reverse_iterator;


	public: /***  The proxy class  ***/

		/**
		 * @class element_proxy
		 * @author Jens Breitbart
		 * @version 0.2
		 * @date 21.07.2007
		 * @brief This class is returned by some functions of @c cupp::vector to determine if an element inside the vector is changed.
		 * @see More effective C++ (Scott Meyers) §30
		 */
		class element_proxy {
			private: /***  Data elements  ***/
				const size_type at_;
				vector<T>& vector_;
				
			public: /***  Constructors & Destructors  ***/
				element_proxy(const size_type at, vector<T>& vector) :
					at_(at), vector_(vector) {}

			public: /***  Operators  ***/
				operator T&() const {
					vector_.update_host();
					return vector_.data_[at_];
				}
				
				const T& get() const {
					vector_.update_host();
					return vector_.data_[at_];
				}
				
				T& get() {
					vector_.update_host();
					vector_.host_changes_ = true;
					return vector_.data_[at_];
				}
				
				element_proxy& operator=(const element_proxy &rhs) {
					rhs.vector_.update_host();
					vector_.update_host();
					vector_.host_changes_ = true;
					vector_.data_[at_]=rhs.vector_.data_[ rhs.at_];
					return *this;
				}
				
				
				element_proxy& operator=(const T& rhs) {
					vector_.update_host();
					vector_.host_changes_ = true;
					vector_.data_[at_] = rhs;
					return *this;
				}

				T* operator&() {
					vector_.update_host();
					return &vector_.data_[at_];
				}
		};
	
		/**
		 * @class iterator
		 * @author Jens Breitbart
		 * @version 0.1
		 * @date 24.07.2007
		 * @brief This class is returned by some functions of @c cupp::vector to determine if an element inside the vector is changed.
		 * @see More effective C++ (Scott Meyers) §30
		 */
		class iterator {
			private: /***  Data elements  ***/
				typename std::vector<T>::iterator i_;
				vector<T>& vector_;
				
			public: /***  Constructors & Destructors  ***/
				iterator(const typename std::vector<T>::iterator i, vector<T>& vector) :
					i_(i), vector_(vector) {}

			public: /***  Operators  ***/
				operator typename std::vector<T>::iterator() const {
					vector_.update_host();
					vector_.host_changes_ = true;
					return i_;
				}

				operator const_iterator() const {
					vector_.update_host();
					return static_cast< const_iterator > (i_);
				}

				const T& operator* () const {
					vector_.update_host();
					return *i_;
				}
				
				T& operator* () {
					vector_.update_host();
					vector_.host_changes_ = true;
					return *i_;
				}
				
				const T& operator-> () const {
					vector_.update_host();
					return *i_;
				}
				
				T& operator-> () {
					vector_.update_host();
					vector_.host_changes_ = true;
					return *i_;
				}

				// prefix
				void operator++ () {
					++i_;
				}

				// postfix
				iterator operator++ (int) {
					const iterator returnee = *this;
					++(*this);
					return returnee;
				}

				// prefix
				void operator-- () {
					--i_;
				}

				// postfix
				iterator operator-- (int) {
					iterator returnee = *this;
					--(*this);
					return returnee;
				}
				
				bool operator!= (const iterator &other) {
					return i_ != other.i_;
				}

				bool operator== (const iterator &other) {
					return i_ == other.i_;
				}
		};
		
		typedef          std::reverse_iterator<iterator>         reverse_iterator;

	public: /***  CONSTRUCTORS AND DESTRUCTORS  ***/
	
		/**
		 * @see @c std::vector
		 */
		vector() : data_(), host_changes_(true), device_changes_(false), ref_invalid_(true), memory_ptr_(0), device_ref_ptr_(0) {}

		/**
		 * @see @c std::vector
		 */
		vector( const vector& c ) : host_changes_(true), device_changes_(false), ref_invalid_(true), memory_ptr_(0), device_ref_ptr_(0) {
			c.update_host();
			data_ = c.data_;
		}

		/**
		 * @see @c std::vector
		 */
		vector( size_type num, const T& val = T() ) : data_(num, val), host_changes_(true), device_changes_(false), ref_invalid_(true), memory_ptr_(0), device_ref_ptr_(0) {}

		/**
		 * @see @c std::vector
		 */
		template <typename input_iterator>
		vector( input_iterator start, input_iterator end ) : data_(start, end), host_changes_(true), device_changes_(false), ref_invalid_(true), memory_ptr_(0), device_ref_ptr_(0) {}

		/**
		 * @see @c std::vector
		 */
		~vector() {
			delete memory_ptr_;
			delete device_ref_ptr_;
		}


	public: /***  OPERATORS  ***/
		/**
		 * @see @c std::vector
		 */
		element_proxy operator[]( size_type index ) {
			return element_proxy (index, *this);
		}

		/**
		 * @see @c std::vector
		 */
		const T& operator[]( size_type index ) const {
			update_host();
			return data_[index];
		}

		/**
		 * @see @c std::vector
		 */
		vector& operator=(const vector& c2) {
			c2.update_host();
			data_ = c2.data_;
			host_changes_ = true;

			// our data have been overwritten ... ignore all old data on the device
			device_changes_ = false;

			return *this;
		}

	public: /***  NORMAL FUNCTIONS  ***/
		/**
		 * @see @c std::vector
		 */
		void assign( size_type num, const T& val ) {
			data_.assign (num, val);
			
			host_changes_ = true;
			// our data have been overwritten ... ignore all old data on the device
			device_changes_ = false;
		}

		/**
		 * @see @c std::vector
		 */
		template <typename input_iterator>
		void assign( input_iterator start, input_iterator end ) {
			data_.assign (start, end);
			
			host_changes_ = true;
			// our data have been overwritten ... ignore all old data on the device
			device_changes_ = false;
		}

		/**
		 * @see @c std::vector
		 */
		element_proxy at( size_type loc ) {
			data_.at(loc);
			return element_proxy (loc, *this);
		}
		
		/**
		 * @see @c std::vector
		 */
		const T& at( size_type loc ) const {
			update_host();
			return data_.at(loc);
		}
		
		/**
		 * @see @c std::vector
		 */
		element_proxy back() {
			return element_proxy(data_.size()-1, *this);
		}
		
		/**
		 * @see @c std::vector
		 */
		const T& back() const {
			update_host();
			return data_.back();
		}

		/**
		 * @see @c std::vector
		 */
		iterator begin() {
			update_host();
			return iterator (data_.begin(), *this);
		}
		
		/**
		 * @see @c std::vector
		 */
		const_iterator begin() const {
			return data_.begin();
		}

		/**
		 * @see @c std::vector
		 */
		size_type capacity() const {
			return data_.capacity();
		}

		/**
		 * @see @c std::vector
		 */
		void clear() {
			data_.clear();
			host_changes_ = true;
			device_changes_ = false;
		}

		/**
		 * @see @c std::vector
		 */
		bool empty() const {
			return data_.empty();
		}

		/**
		 * @see @c std::vector
		 */
		iterator end() {
			update_host();
			return iterator (data_.end(), *this);
		}
		
		/**
		 * @see @c std::vector
		 */
		const_iterator end() const {
			update_host();
			return data_.end();
		}
		
		/**
		 * @see @c std::vector
		 */
		iterator erase( iterator loc ) {
			update_host();
			host_changes_ = true;
			return iterator (data_.erase(loc), *this);
		}
		
		/**
		 * @see @c std::vector
		 */
		iterator erase( iterator start, iterator end ) {
			update_host();
			host_changes_ = true;
			return iterator (data_.erase(start, end), *this);
		}
		
		/**
		 * @see @c std::vector
		 */
		element_proxy front() {
			return element_proxy(0, *this);
		}
		
		/**
		 * @see @c std::vector
		 */
		const T& front() const {
			update_host();
			return data_.front();
		}

		/**
		 * @see @c std::vector
		 */
		iterator insert( iterator loc, const T& val ) {
			update_host();
			host_changes_ = true;
			return iterator (data_.insert(loc, val), *this);
		}
		
		/**
		 * @see @c std::vector
		 */
		void insert( iterator loc, size_type num, const T& val ) {
			update_host();
			host_changes_ = true;
			data_.insert(loc, num, val);
		}

		/**
		 * @see @c std::vector
		 */
		template <typename input_iterator>
		void insert( iterator loc, input_iterator start, input_iterator end ) {
			update_host();
			host_changes_ = true;
			data_.insert(loc, start, end);
		}

		/**
		 * @see @c std::vector
		 */
		size_type max_size() const {
			return data_.max_size();
		}

		/**
		 * @see @c std::vector
		 */
		void pop_back() {
			update_host();
			data_.pop_back();
			host_changes_ = true;
		}

		/**
		 * @see @c std::vector
		 */
		void push_back( const T& val ) {
			update_host();
			data_.push_back(val);
			host_changes_ = true;
		}

		/**
		 * @see @c std::vector
		 */
		reverse_iterator rbegin() {
			update_host();
			host_changes_ = true;
			return data_.rbegin();
		}

		/**
		 * @see @c std::vector
		 */
		const_reverse_iterator rbegin() const {
			update_host();
			return data_.rbegin();
		}

		/**
		 * @see @c std::vector
		 */
		reverse_iterator rend() {
			update_host();
			host_changes_ = true;
			return data_.rend();
		}
		
		/**
		 * @see @c std::vector
		 */
		const_reverse_iterator rend() const {
			update_host();
			return data_.rend();
		}

		/**
		 * @see @c std::vector
		 */
		void reserve( size_type size ) {
			data_.reserve(size);
		}

		/**
		 * @see @c std::vector
		 */
		void resize( size_type num, const T& val = T() ) {
			update_host();
			data_.resize (num, val);
			host_changes_ = true;
		}

		/**
		 * @see @c std::vector
		 */
		size_type size() const {
			return data_.size();
		}

		/**
		 * @see @c std::vector
		 */
		void swap( vector<T>& from ) {
			update_host();
			
			from.update_host();
			data_.swap(from.data);
			
			host_changes_ = true;
			from.host_changes_ = true;
		}

	public: /*** CuPP kernel call traits implementation ***/
		/**
		 * @brief This function is called by the kernel_call_traits
		 * @return The device type for our vector
		 */
		device_type transform (const device &d) {
			update_device(d);

			device_type temp;
			temp.set_size ( size());
			temp.set_device_pointer (memory_ptr_->cuda_pointer().get());
			return temp;
		}

		/**
		 * @brief This function is called by the kernel_call_traits
		 * @return A on the device useable vector reference
		 */
		device_reference< device_type > get_device_reference(const device &d) {

			update_device(d);

			if (ref_invalid_) {
				delete device_ref_ptr_;

				device_ref_ptr_ = new device_reference<device_type> (d, transform(d));
				
				ref_invalid_ = false;
			}

			assert (device_ref_ptr_!=0);

			return *device_ref_ptr_;
		}

		/**
		 * @brief This function is called by the kernel_call_traits
		 */
		void dirty (device_reference< device_type > device_copy) {
			UNUSED_PARAMETER(device_copy);
			
			device_changes_ = true;
		}

		/*void update (const device_type &value) {
			UNUSED_PARAMETER(value);

			device_changes_ = true;
		}*/

		vector< T >&  operator= (const device_type& value) {
			UNUSED_PARAMETER(value);
			
			device_changes_ = true;
			return *this;
		}

		/**
		 * If there is newer data on the device, this function will update the host data with it
		 */
		void update_host() const {
			if (device_changes_) {
				assert(!host_changes_);
				assert(memory_ptr_!=0);

				std::vector< T_device_type > temp( data_.size() );
				
				memory_ptr_ -> copy_to_host (&temp[0]);

				for (std::size_t i = 0; i<temp.size(); ++i) {
					data_[i] = temp[i];
				}
				
				
				device_changes_ = false;
			}
		}

		/**
		 * If there is newer data on the host, this function will update the device data with it
		 */
		void update_device(const device &d) {
			// changes on the host side or we are executed on a new device
			if (host_changes_ || d.id() != device_id_) {

				std::vector< T_device_type > temp;
				/// @todo use a std algo
				for (typename std::vector< T >::iterator it = data_.begin(); it != data_.end(); ++it) {
					temp.push_back ( kernel_call_traits< T, T_device_type >::transform (d, *it) );
				}
				
				// we don't have any space on the device, or the we have more/less space on the device than we neeed
				// or we are executed on a new device
				if (memory_ptr_ == 0 || memory_ptr_ -> size()!=data_.size() || d.id() != device_id_) {
					// free the memory
					delete memory_ptr_;
					
					// get new memory and copy the data
					memory_ptr_ = new memory1d<T_device_type>(d, &temp[0], temp.size() );

					// we need to create a new proxy because our memory has a new address
					ref_invalid_ = true;
				} else {
					// copy the data to the device
					memory_ptr_ -> copy_to_device (&temp[0]);
				}

				device_id_ = d.id();
				host_changes_ = false;
			}
		}
		

	private:
		/**
		 * Our real vector :-)
		 */
		mutable std::vector<T> data_;
		
		/**
		 * true means, data has been changed on the host and data on the device is out of date
		 */
		mutable bool host_changes_;

		/**
		 * true means, data has been changed on the device and data on the host is out of date
		 */
		mutable bool device_changes_;

		/**
		 * True means we have to recreate our proxy
		 */
		mutable bool ref_invalid_;

		/**
		 * Our device memory :-)
		 */
		mutable memory1d< T_device_type > *memory_ptr_;

		/**
		 * The proxy on our device
		 */
		mutable device_reference<device_type> *device_ref_ptr_;

		/**
		 * A pointer to the device on which our data is stored
		 */
		mutable device::id_t device_id_;
}; // class vector


template <typename T1, typename T2>
bool operator==(const vector<T1>& c1, const vector<T2>& c2) {
	c1.update_host();
	c2.update_host();
	return c1.data_ == c2.data_;
}

template <typename T1, typename T2>
bool operator!=(const vector<T1>& c1, const vector<T2>& c2) {
	return !(c1 == c2);
}

template <typename T1, typename T2>
bool operator<(const vector<T1>& c1, const vector<T2>& c2) {
	c1.update_host();
	c2.update_host();
	return c1.data_ < c2.data_; 
}

template <typename T1, typename T2>
bool operator>(const vector<T1>& c1, const vector<T2>& c2) {
	c1.update_host();
	c2.update_host();
	return c1.data_ > c2.data_;
}

template <typename T1, typename T2>
bool operator<=(const vector<T1>& c1, const vector<T2>& c2) {
	return !(c1 > c2);
}

template <typename T1, typename T2>
bool operator>=(const vector<T1>& c1, const vector<T2>& c2) {
	return !(c1 < c2);
}


} // namespace cupp

#endif

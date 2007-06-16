/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_memory1d_H
#define CUPP_memory1d_H

// Include std::size_t
#include <cstddef>

#include "cupp_common.h"



namespace cupp {

/// @code_review Please put the public members first and then the private ones.
/**
 * @class memory1d
 * @author Jens Breitbart
 * @version 0.1
 * @date 13.06.2007
 * @brief A pointer to linear 1-dimensional memory.
 * @warning You have to free this memory, or you will have a ressource leak!
 *
 * asdvasdvasdv
 */
template <typename T>
class memory1d {
	private: /***  INTERNAL DATA  ***/
		const T* pointer_;
		/// @code_review Should we remove @c size_ and use @c cudaGetSymbolSize instead?
		const std::size_t size_;
		
	private: /***  CONSTRUCTORS & DESTRUCTORS ***/
		memory1d() {};
		memory1d(const T* pointer, const std::size_t size) :
			pointer_(pointer), size_(size) {}
	public:
		/// @code_review Do typedefs work with nvcc?
		typedef std::size_t size_type;
		typedef T value_type;
		typedef T* pointer;
		
		
		/**
		 * @brief Copy constructor
		 * @param copy
		 * @todo Should we test if the pointer is used in the correct context?
		 * @todo Do we need a public copy-constructor?
		 */
		memory1d(const memory1D& copy) : pointer_(copy.pointer_), size_(copy.size_) {}

	public:
		
		/**
		 * @return How many elements we can store on the device
		 * @platform Device
		 */
		///  @code_review Should we define a special header that defines macros like
		CUPP_HOST CUPP_DEVICE
		size_type size() const {
			return size_;
		}

	public: /***  GPU FUNCTIONS  ***/
	#if defined(__CUDACC__)
		/**
		 * @brief Cast the memory1D into a simple pointer
		 * @warning Only available on the GPU.
		 */
		CUPP_DEVICE
		operator T*() const {
			return pointer_;
		}
	#endif // defined(__CUDACC__)

	/// @code_review If friendship must be delcared then the class should be in the same file as its
	///              friend.
	friend cupp::device;
}; // class memory1d

} // namespace cupp

#endif

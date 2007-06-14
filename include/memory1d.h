/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */


#ifndef CUPP_device_H
#define CUPP_device_H

namespace cupp {

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
		const std::size_t size_;
		
	private: /***  CONSTRUCTORS & DESTRUCTORS ***/
		memory1d() {};
		memory1d(const T* pointer, const std::size_t size) :
			pointer_(pointer), size_(size) {}
	public:
		/**
		 * @brief Copy constructor
		 * @param copy
		 * @todo Should we test if the pointer is used in the correct context?
		 * @todo Do we need a public copy-constructor?
		 */
		memory1d(const memory1D &copy) : pointer_(copy.pointer_), size_(copy.size_) {}

	public:
		
		/**
		 * @return How many elements we can store on the device
		 */
		#if defined(__CUDACC__)
		__host__ __device__
		#endif
		std::size_t size() const {
			return size_;
		}

	public: /***  GPU FUNCTIONS  ***/
	#if defined(__CUDACC__)
		/**
		 * @brief Cast the memory1D into a simple pointer
		 * @warning Only available on the GPU.
		 */
		__device__
		operator T*() const {
			return pointer_;
		}
	#endif // defined(__CUDACC__)

	friend cupp::device;
}; // class memory1d

} // namespace cupp

#endif
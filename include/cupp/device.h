/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_device_H
#define CUPP_device_H

#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif

namespace cupp {


/**
 * @class device
 * @author Jens Breitbart
 * @version 0.2
 * @date 03.08.2007
 * @platform Host only!
 * @brief This class is a handle to a CUDA device. You need it to allocate data, run kernel, ...
 * @warning If you destroy your device object, all data located on it will be destroyed.
 */
class device {
	public:
		typedef int id_t;
	
	public: /***  CONSTRUCTORS & DESTRUCTORS ***/
		/**
		 * @brief Generates a default device with no special requirements
		 */
		device();

		/**
		 * @brief Cleans up all ressources associated with the device
		 */
		~device();

	public:
		/**
		 * @brief This functions blocks until all requested tasks/kernels have been completed
		 */
		void sync() const;
	
	public: /***  UTILITY FUNCTIONS  ***/
		/**
		 * @return the number of available devices
		 */
		static int device_count();

		int spes() { return number_of_spes; }

		void set_spes( const int i) { number_of_spes = i; }

	private:
		/**
		 * @brief This is the real constructor.
		 * @param major The requested major rev. number; pass -1 to ignore it
		 * @param minor The requested minor rev. number; pass -1 to ignore it
		 * @param name  The requested device name; pass 0 to ignore it
		 */
		void real_constructor(const int major, const int minor, const char* name);

	private:
		int number_of_spes;
}; // class device

} // namespace cupp

#endif

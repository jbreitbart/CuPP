/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef CUPP_device_H
#define CUPP_device_H

#if defined(__CUDACC__)
#error Not compatible with CUDA. Don't compile with nvcc.
#endif

namespace cupp {


/**
 * @class device
 * @author Jens Breitbart
 * @version 0.1
 * @date 13.06.2007
 * @platform Host only!
 * @warning 
 * @brief asdvasd
 *
 * asdvasdvasdv
 */
class device {
	public: /***  CONSTRUCTORS & DESTRUCTORS ***/
		/**
		 * @brief Generates a default device with no special requirements
		 */
		device();

		/**
		 * @brief Generates a device with @a major revision number
		 * @param major The requested major revision number of the device
		 */
		explicit device (const int major);
		
		/**
		 * @brief Generates a device with @a major and @a minor revision number
		 * @param major The requested major revision number of the device
		 * @param minor The requested minor revision number of the device
		 */
		explicit device (const int major, const int minor);

	
	public: /*** UTILITY FUNCTIONS ***/
		/**
		 * @return the number of available devices
		 */
		static int device_count();
	
	private:	
		/**
		 * @brief This is the real constructor.
		 * @param major The requested major rev. number; pass -1 to ignore it
		 * @param minor The requested minor rev. number; pass -1 to ignore it
		 * @param name  The requested device name; pass 0 to ignore it
		 */
		void real_constructor(const int major, const int minor, const char* name);
}; // class device

} // namespace cupp

#endif

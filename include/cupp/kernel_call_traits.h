/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_kernel_call_traits_H
#define CUPP_kernel_call_traits_H

#include "cupp/runtime.h"
#include "cupp/device.h"
#include "cupp/device_reference.h"

#include <iostream>

#include <boost/utility.hpp>
#include <boost/type_traits.hpp>

namespace cupp {

namespace impl {

using boost::enable_if;
using boost::disable_if;

/**
 * @class has_member_transform
 * @author Jens Breitbart
 * @version 0.1
 * @date 03.01.2008
 * @brief Helper class used to detect if a type defines the transform member function
 */
template<class T> struct has_member_transform;

template<class R, class C>
class has_member_transform<R C::*> {
	private:
		typedef char one;
		typedef char (&two)[2];

		template<R C::*> struct helper;

		// if function transform is defined in T:
		template<class T> static one check(helper<&T::transform>*);
		//else:
		template<class T> static two check(...);

	public:
		enum {value = (sizeof(check<C>(0)) == sizeof(char))};
};

/**
 * @class transform_caller
 * @author Jens Breitbart
 * @version 0.1
 * @date 03.01.2008
 * @brief Helper class used to call the transform function
 */
template <typename device_type>
struct transform_caller {

	/// instantiated if host_type::transform() exists
	template <typename host_type>
	static device_type call( const device &d, host_type& that,
	                         typename enable_if <
	                           has_member_transform <device_type (host_type::*)(const device&) >
	                         >::type* = 0 ) {
		return that.transform (d);
	}

	/// instantiated if that.transform() does NOT exists
	template <typename host_type>
	static device_type call( const device &d, host_type& that,
	                          typename disable_if <
	                            has_member_transform<device_type (host_type::*)(const device&) >
	                          >::type* = 0 ) {
		UNUSED_PARAMETER(d);
		return static_cast<device_type>(that);
	}

	/// instantiated if 'host_type' is a POD
	template <typename host_type>
	static device_type& call( const device &d, host_type& that, typename boost::enable_if< boost::is_POD< host_type > >::type* = 0) {
		UNUSED_PARAMETER(d);
		return that;
	}

};



/**
 * @class has_member_get_device_reference
 * @author Jens Breitbart
 * @version 0.1
 * @date 03.01.2008
 * @brief Helper class used to detect if a type defines the get_device_reference member function
 */
template<class T> struct has_member_get_device_ref;

template<class R, class C>
class has_member_get_device_ref<R C::*> {
	private:
		typedef char one;
		typedef char (&two)[2];

		template<R C::*> struct helper;

		// if function get_device_reference() is defined in T:
		template<class T> static one check(helper<&T::get_device_reference>*);
		//else:
		template<class T> static two check(...);

	public:
		enum {value = (sizeof(check<C>(0)) == sizeof(char))};
};

/**
 * @class get_device_ref_caller
 * @author Jens Breitbart
 * @version 0.1
 * @date 03.01.2008
 * @brief Helper class used to call the get_device_reference function
 */
template <typename device_type>
struct get_device_ref_caller {

	/// instantiated if host_type::get_device_reference() exists
	template <typename host_type>
	static device_reference<device_type> call( const device &d, host_type& that,
	                         typename enable_if <
	                           has_member_get_device_ref <device_reference<device_type> (host_type::*)(const device&)  >
	                         >::type* = 0 ) {
		return that.get_device_reference (d);
	}

	/// instantiated if host_type::get_device_reference() does NOT exists
	template <typename host_type>
	static device_reference<device_type> call( const device &d, host_type& that,
	                          typename disable_if <
	                            has_member_get_device_ref <
				       device_reference<device_type> (host_type::*)(const device&) 
				                              >
	                          >::type* = 0 )


				  {
		return cupp::device_reference < device_type > (d, transform_caller<device_type>::call(d, that) );
	}

	/// instantiated if 'host_type' is a POD
	template <typename host_type>
	static device_reference<device_type> call( const device &d, host_type& that, typename boost::enable_if< boost::is_POD< host_type > >::type* = 0) {
		return device_reference<device_type> (d, that);
	}
};



/**
 * @class has_member_get_device_reference
 * @author Jens Breitbart
 * @version 0.1
 * @date 03.01.2008
 * @brief Helper class used to detect if a type defines the dirty member function
 */
template<class T> struct has_member_dirty;

template<class R, class C>
class has_member_dirty<R C::*> {
	private:
		typedef char one;
		typedef char (&two)[2];

		template<R C::*> struct helper;

		// if function dirty() is defined in T:
		template<class T> static one check(helper<&T::dirty>*);
		//else:
		template<class T> static two check(...);

	public:
		enum {value = (sizeof(check<C>(0)) == sizeof(char))};
};

/**
 * @class dirty_caller
 * @author Jens Breitbart
 * @version 0.1
 * @date 03.01.2008
 * @brief Helper class used to call the dirty function
 */
template <typename device_type>
struct dirty_caller {

	/// instantiated if host_type::dirty() exists
	template <typename host_type>
	static void call( host_type& that, device_reference<device_type> device_ref,
	                         typename enable_if <
	                           has_member_dirty < void (host_type::*)(device_reference<device_type>) >
	                         >::type* = 0 ) {
		that.dirty(device_ref);
	}

	/// instantiated if host_type::dirty() does NOT exists
	template <typename host_type>
	static void call( host_type& that, device_reference<device_type> device_ref,
	                          typename disable_if <
	                            has_member_dirty < void (host_type::*)(device_reference<device_type>) >
	                          >::type* = 0 ) {
		that = static_cast<host_type>(device_ref.get());
	}

	/// instantiated if 'host_type' is a POD
	template <typename host_type>
	static void call( host_type& that, device_reference<device_type> device_ref,
	                  typename boost::enable_if< boost::is_POD< host_type > >::type* = 0) {
		that = device_ref.get();
	}
};

}


/**
 * @class kernel_call_traits
 * @author Jens Breitbart
 * @version 0.5
 * @date 03.01.2008
 * @brief These traits define the behavior of what happens when a kernel is called.
 */
template <typename host_type, typename device_type>
struct kernel_call_traits {

	/**
	 * Transforms the host type to the device type
	 * @param d The device the kernel will be executed on
	 * @param that The object that is about to be passed to the kernel
	 * @note This function is called when you pass a parameter by value to a kernel.
	 */
	static device_type transform (const device &d, host_type& that) {
		using namespace impl;
		return transform_caller<device_type>::call(d, const_cast<host_type&>(that) );
	}

	/**
	 * Creates a device reference to be passed to the kernel
	 * @param d The device the kernel will be executed on
	 * @param that The object that is about to be passed to the kernel
	 * @note This function is called when you pass a parameter by reference to a kernel.
	 */
	static device_reference<device_type> get_device_reference (const device &d, host_type& that) {
		using namespace impl;
		return get_device_ref_caller<device_type>::call (d, that);
	}

	/**
	 * This function is called when the value may have been changed on the device.
	 * @param that The host representation of your data
	 * @param device_copy The pointer you created with @a get_device_based_device_copy
	 * @note This function is only called if you pass a parameter by non-const reference to a kernel.
	 */
	static void dirty (host_type& that, device_reference<device_type> device_ref) {
		using namespace impl;
		dirty_caller<device_type>::call(that, device_ref);
	}
};


} // cupp

#endif //CUPP_kernel_call_traits_H

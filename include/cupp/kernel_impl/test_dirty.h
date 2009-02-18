/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_test_dirty_H
#define CUPP_KERNEL_IMPL_test_dirty_H

// STD
#include <vector>

// BOOST
#include <boost/type_traits.hpp>

namespace cupp {
namespace kernel_impl {

/**
 * @class test_dirty
 * @author Jens Breitbart
 * @version 0.2
 * @date 13.02.2008
 * @brief Use by cupp::kernel_impl::kernel_launcher_impl to introspec the __global__ function and to detetmine which arguments of the functions are non-const references
 * @warning The is very strictly tied to kernel_launcher_impl; I would not expect any usage beyond it.
 * @note This is a friend class of kernel_launcher_impl, but not defined into the same header to make it easier to extend it and to keep it clear.
 */

template <int arirty>
class test_dirty {
	/**
	 * @param F is the type of the __global__ function to be introspected
	 * @return The vector will have the size of the arity of the __global__ function. @a true at pos 0 means the first parameter is passed as non-const reference.
	 * @warning If you come here with a compiler warning about dirty beeing private, you probably hit the maximum number of parameters supported by @c cupp::kernel. If you need more parameters, just extend @c cupp::kernel_imp::test_dirty, @c cupp::kernel::operator() and @c cupp::kernel_imp::real_setup_argument ... mostly just a copy and paste work.
	 */
	template <typename F>
	static std::vector<bool> dirty ();
};

#define CUPP_TEST_DIRTY_HEADER(a,b) \
template <> \
class test_dirty<a> { \
	template <typename F> \
	static std::vector<bool> dirty (); \
	\
	template <typename T> \
	friend class kernel_launcher_impl; \
	\
	template <typename T> \
	friend std::vector<bool> test_dirty<b>::dirty(); \
};


template <>
class test_dirty<16> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
};

//CUPP_TEST_DIRTY_HEADER(16,17)
CUPP_TEST_DIRTY_HEADER(15,16)
CUPP_TEST_DIRTY_HEADER(14,15)
CUPP_TEST_DIRTY_HEADER(13,14)
CUPP_TEST_DIRTY_HEADER(12,13)
CUPP_TEST_DIRTY_HEADER(11,12)
CUPP_TEST_DIRTY_HEADER(10,11)
CUPP_TEST_DIRTY_HEADER(9,10)
CUPP_TEST_DIRTY_HEADER(8,9)
CUPP_TEST_DIRTY_HEADER(7,8)
CUPP_TEST_DIRTY_HEADER(6,7)
CUPP_TEST_DIRTY_HEADER(5,6)
CUPP_TEST_DIRTY_HEADER(4,5)
CUPP_TEST_DIRTY_HEADER(3,4)
CUPP_TEST_DIRTY_HEADER(2,3)
CUPP_TEST_DIRTY_HEADER(1,2)
CUPP_TEST_DIRTY_HEADER(0,1)

#undef CUPP_TEST_DIRTY_HEADER

#if 0
template <>
class test_dirty<8> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	template <typename T>
	friend std::vector<bool> test_dirty<9>::dirty();
};

template <>
class test_dirty<7> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	template <typename T>
	friend std::vector<bool> test_dirty<8>::dirty();
};

template <>
class test_dirty<6> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	template <typename T>
	friend std::vector<bool> test_dirty<7>::dirty();
};

template <>
class test_dirty<5> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	template <typename T>
	friend std::vector<bool> test_dirty<6>::dirty();
};

template <>
class test_dirty<4> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	template <typename T>
	friend std::vector<bool> test_dirty<5>::dirty();
};

template <>
class test_dirty<3> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	template <typename T>
	friend std::vector<bool> test_dirty<4>::dirty();
};

template <>
class test_dirty<2> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	template <typename T>
	friend std::vector<bool> test_dirty<3>::dirty();
};

template <>
class test_dirty<1> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	template <typename T>
	friend std::vector<bool> test_dirty<2>::dirty();
};

template <>
class test_dirty<0> {
	template <typename F>
	static std::vector<bool> dirty ();
	
	template <typename T>
	friend class kernel_launcher_impl;
	
	template <typename T>
	friend std::vector<bool> test_dirty<1>::dirty();
};
#endif

/*** IMPLEMENTATION ***/

/**
 * @returns @a true if ARG is a non-const reference
 */
template <typename ARG>
inline bool check_arg() {
	if (boost::is_reference<ARG>::value && !is_second_level_const<ARG>::value) {
		return true;
	}
	
	return false;
}

template <typename F>
std::vector<bool> test_dirty<0>::dirty () {
	return std::vector<bool>();
}

#define CUPP_TEST_DIRTY(a,b) \
template <typename F> \
std::vector<bool> test_dirty<a>::dirty () { \
	typedef typename boost::function_traits<F>::arg##a##_type ARG; \
	\
	std::vector< bool > tmp(test_dirty<b>::dirty<F>()); \
	tmp.push_back (check_arg<ARG>()); \
	return tmp; \
}

CUPP_TEST_DIRTY(1,0)
CUPP_TEST_DIRTY(2,1)
CUPP_TEST_DIRTY(3,2)
CUPP_TEST_DIRTY(4,3)
CUPP_TEST_DIRTY(5,4)
CUPP_TEST_DIRTY(6,5)
CUPP_TEST_DIRTY(7,6)
CUPP_TEST_DIRTY(8,7)
CUPP_TEST_DIRTY(9,8)
CUPP_TEST_DIRTY(10,9)
CUPP_TEST_DIRTY(11,10)
CUPP_TEST_DIRTY(12,11)
CUPP_TEST_DIRTY(13,12)
CUPP_TEST_DIRTY(14,13)
CUPP_TEST_DIRTY(15,14)
CUPP_TEST_DIRTY(16,15)

#undef CUPP_TEST_DIRTY

#if 0
template <typename F>
std::vector<bool> test_dirty<1>::dirty () {
	typedef typename boost::function_traits<F>::arg1_type ARG;
	
	std::vector< bool > tmp(test_dirty<0>::dirty<F>());
	tmp.push_back (check_arg<ARG>());
	return tmp;
}

template <typename F>
std::vector<bool> test_dirty<2>::dirty () {
	typedef typename boost::function_traits<F>::arg2_type ARG;
	
	std::vector< bool > tmp(test_dirty<1>::dirty<F>());
	tmp.push_back (check_arg<ARG>());
	return tmp;
}

template <typename F>
std::vector<bool> test_dirty<3>::dirty () {
	typedef typename boost::function_traits<F>::arg3_type ARG;
	
	std::vector< bool > tmp(test_dirty<2>::dirty<F>() );
	tmp.push_back (check_arg<ARG>());
	return tmp;
}

template <typename F>
std::vector<bool> test_dirty<4>::dirty () {
	typedef typename boost::function_traits<F>::arg4_type ARG;
	
	std::vector< bool > tmp(test_dirty<3>::dirty<F>() );
	tmp.push_back (check_arg<ARG>());
	return tmp;
}

template <typename F>
std::vector<bool> test_dirty<5>::dirty () {
	typedef typename boost::function_traits<F>::arg5_type ARG;
	
	std::vector< bool > tmp(test_dirty<4>::dirty<F>() );
	tmp.push_back (check_arg<ARG>());
	return tmp;
}

template <typename F>
std::vector<bool> test_dirty<6>::dirty () {
	typedef typename boost::function_traits<F>::arg6_type ARG;
	
	std::vector< bool > tmp(test_dirty<5>::dirty<F>() );
	tmp.push_back (check_arg<ARG>());
	return tmp;
}

template <typename F>
std::vector<bool> test_dirty<7>::dirty () {
	typedef typename boost::function_traits<F>::arg7_type ARG;
	
	std::vector< bool > tmp(test_dirty<6>::dirty<F>() );
	tmp.push_back (check_arg<ARG>());
	return tmp;
}

template <typename F>
std::vector<bool> test_dirty<8>::dirty () {
	typedef typename boost::function_traits<F>::arg8_type ARG;
	
	std::vector< bool > tmp(test_dirty<7>::dirty<F>() );
	tmp.push_back (check_arg<ARG>());
	return tmp;
}
#endif

} // kernel_impl
} // cupp

#endif //CUPP_KERNEL_IMPL_real_setup_argument_H

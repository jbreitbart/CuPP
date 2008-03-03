/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_KERNEL_IMPL_is_second_level_const_H
#define CUPP_KERNEL_IMPL_is_second_level_const_H

namespace cupp {
namespace kernel_impl {

/**
 * @class is_second_level_const
 * @author Jens Breitbart with some help from comp.lang.c++.moderated
 * @version 0.1
 * @date 19.07.2007
 * @brief Determine if the passed parameter is a const reference or a const pointer
 * @example is_second_level_const <int const*>::value == is_second_level_const <const int*>::value == true;
 * @example is_second_level_const <int* const>::value == false; this is top level const (use boost if you need this)
 */

template <typename T>
class is_second_level_const {
public:
	enum {value = false};
};

template <typename T>
class is_second_level_const<const T&> {
public:
	enum {value = true};
};

template <typename T>
class is_second_level_const<const T*> {
public:
	enum {value = true};
};

}
}

#endif //CUPP_KERNEL_IMPL_is_second_level_const_H

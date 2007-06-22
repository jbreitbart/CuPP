/* The following code example is taken from the book
 * "C++ Templates - The Complete Guide"
 * by David Vandevoorde and Nicolai M. Josuttis, Addison-Wesley, 2002
 *
 * (C) Copyright David Vandevoorde and Nicolai M. Josuttis 2002.
 * Permission to copy, use, modify, sell and distribute this software
 * is granted provided this copyright notice appears in all copies.
 * This software is provided "as is" without express or implied
 * warranty, and with no claim as to its suitability for any purpose.
 */
#ifndef FORWARD_HPP
#define FORWARD_HPP


namespace cupp {
namespace kernel_magic {

// ForwardParamT<T>::Type is
// - constant reference for class types
// - plain type for almost all other types
// - a dummy type (Unused) for type void
template<typename T>
class ForwardParamT {
  public:
    typedef T const & Type;
};

template<>
class ForwardParamT<void> {
  private:
    class Unused {};
  public:
    typedef Unused Type;
};

}
}

#endif // FORWARD_HPP

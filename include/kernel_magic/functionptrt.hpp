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

namespace cupp {
namespace kernel_magic {

// primary template handles maximum number of parameters:
template<typename P1 = void,
         typename P2 = void,
         typename P3 = void>
class FunctionPtrT {
  public:
    enum { NumParams = 3 };
    typedef void (*Type)(P1,P2,P3);
};

// partial specialization for two parameters:
template<typename P1,
         typename P2>
class FunctionPtrT<P1, P2, void> {
  public:
    enum { NumParams = 2 };
    typedef void (*Type)(P1,P2);
};

// partial specialization for one parameter:
template<typename P1>
class FunctionPtrT<P1, void, void> {
  public:
    enum { NumParams = 1 };
    typedef void (*Type)(P1);
};

// partial specialization for no parameters:
template<>
class FunctionPtrT<void, void, void> {
  public:
    enum { NumParams = 0 };
    typedef void (*Type)();
};

}
}

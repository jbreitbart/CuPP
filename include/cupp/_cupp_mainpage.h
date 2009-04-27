/**
 * \mainpage
 * 
 * \section intro Introduction
 * The CuPP framework has been developed to ease the integration of CUDA into C++
 * applications. It enhances CUDA by offering automatic device/host memory management, data structures
 * and a special kernel call semantic, similar to call by reference as it is known from the C++ realm.
 * 
 * \section HP Homepage
 * - CuPP homepage: http://www.plm.eecs.uni-kassel.de/plm/index.php?id=cupp
 * - Documentation: http://cupp.gpuified.de/
 * - Google group: http://groups.google.com/group/cupp
 * 
 * \section rel_notes Release Notes
 * <a href="static/cupp_v0.1.2.tar.gz">Download version 0.1.2 of the CuPP framework</a>
 * 
 * This release of CuPP is only tested on Ubuntu Linux (32bit),
 * but it is expected to work well on other linux flavors. Windows is untested and not expected to work
 * correctly, but maybe in an upcoming release.
 * 
 * The downloadable file contains:
 * - the CuPP framework itself
 * - simple example applications demonstrating the usage of CuPP
 * 
 * \section start Getting Started
 * \subsection req Requirements
 * The CuPP framework requires the following software to be installed:
 * - <a href="http://www.cmake.org">CMake</a>, to generate the build script
 * - <a href="http://www.nvidia.com/object/cuda_home.html">CUDA</a> version 2.0
 * - <a href="http://www.boost.org">Boost libraries</a>
 * 
 * \subsection build Building the CuPP framework
 * Detail instructions of how to build CuPP are included in the download file. If you have any problems
 * join us at our google group.
 * 
 * \subsection example Examples
 * Examples are included in the download file in the subdirectory 'examples'.
 * 
 * \subsection limit Known limitation
 * - The number of parameters that can be passed to a kernel is limited by the function arity supported
 *   by function_traits of Boost.TypeTraits.
 *
 * \section overview Overview
 * <img src="static/overview.png"/>
 * The CuPP framework consists of 5 highly interwoven parts of which some replace the existing CUDA counterparts
 * whereas others offer new functionalities.
 * - <b>Device management</b> \n
 *   Device management is not done implicit by associating a thread with a
 *   device as it is done by CUDA. Instead, the developer is forced to create a
 *   device handle (cupp::device), which is passed to all CuPP functions using
 *   the device, e.g. kernel calls and memory allocation.
 * - <b>Memory management</b> \n
 *   Two different memory management concepts are available.
 *   - One is identical to the one offered by CUDA, unless that
 *     exceptions are thrown when an error occurs instead of returning an error code.
 *     To ease the development with this basic approach, a boost library-compliant
 *     shared pointer for global memory is supplied.
 *   - The second type of memory management uses a class called cupp::memory1d.
 *     Objects of this class represent a linear block of global memory. The memory is
 *     allocated when the object is created and freed when the object is destroyed.
 *     Data can be transferred to the memory from any data structure supporting iterators.
 * - <b>C++ kernel call</b> \n
 *   The CuPP kernel call is implemented by a C++ functor (cupp::kernel), which
 *   adds a call by reference like semantic to basic CUDA kernel calls. This can be used
 *   to pass datastructures like cupp::vector to a kernel, so the device can modify them.
 * - <b>Support for classes</b> \n
 *   Using a technique called "type transformations" generic C++ classes can easily be transferred to
 *   and from device memory.
 * - <b>Data structures</b> \n
 *   Currently only a std::vector wrapper offering automatic memory
 *   management is supplied. This class also implements a feature called lazy memory copying, to
 *   minimize any memory transfers between device and host memory. Currently no other datastructures are
 *   supplied, but can be added with ease.
 *
 * A document describing all functionalities in detail, can be found in the references section.
 * 
 *
 * 
 * \section ref References
 * An detail description of the CuPP framework can be found in:
 * - J. Breitbart. A framework for easy CUDA integration in C++ applications.<br/>
 *   Diplomarbeit, University of Kassel, 2008.<br/>
 *   http://www.plm.eecs.uni-kassel.de/plm/fileadmin/pm/publications/breitbart/framework_for_easy_cuda_integration_c___applications.pdf
 * 
 * \section ex Example
 * - J. Breitbart. Case studies on GPU usage and data structure design.<br/>
 *   Master Thesis, University of Kassel, 2008.<br/>
 *   http://www.plm.eecs.uni-kassel.de/plm/fileadmin/pm/publications/breitbart/case_studies_on_gpu_usage_and_data_structure_design.pdf
 * 
 * \section credit Credits
 * 
 * \subsection developer Developers in alphabetical order
 * - <a href="http://www.plm.eecs.uni-kassel.de/plm/index.php?id=breitbart">Jens Breitbart</a>
 * - <a href="http://www.plm.eecs.uni-kassel.de/plm/index.php?id=bknafla">Björn Knafla</a>,
 * 
 * \subsection ack Acknowledgments fly out to
 * - <a href="http://www.sci.utah.edu/~abe/">Abe Stephens</a> for his very usefull CUDA CMake
 *   <a href="http://www.sci.utah.edu/~abe/FindCuda.html">script</a>.
 * - Everyone working on the <a href="http://www.boost.org/">Boost Libraries</a> for the
 *   powerful yet easy to use libraries.
 * - comp.lang.c++.moderated for some enlightening C++ code snippets.
 * 
 * \section licences Software License
 * The CuPP framework is licenced under the BSD licence. The detailed license is
 * included in the downloadable file.
 */


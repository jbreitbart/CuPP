/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_CELL_cuda_stub_H
#define CUPP_CELL_cuda_stub_H


#if defined(__CUDACC__)
#error "Not compatible with CUDA. Don't compile with nvcc."
#endif


// CUPP

// BOOST
#include <boost/type_traits.hpp>

namespace cupp {

namespace cell {

	struct dim3 {
		int x;
		int y;
		int z;
		
		dim3 () : x(0), y(0), z(0) {}
		dim3 (int x_) : x(x_), y(0), z(0) {}
		dim3 (int x_, int y_) : x(x_), y(y_), z(0) {}
		dim3 (int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
	};



} // cell

} // cupp

#endif // CUPP_CELL_cuda_stub_H

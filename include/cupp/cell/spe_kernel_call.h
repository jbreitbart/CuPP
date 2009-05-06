/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_CELL_spe_kernel_call_H
#define CUPP_CELL_spe_kernel_call_H

// CUPP
#include "cupp/cell/cuda_stub.h"

// BOOST
#include <boost/type_traits.hpp>

namespace cupp {

namespace cell {


	template <typename F_>
	inline void call_kernel (F_ kernel_ptr, dim3 &gridDim, dim3 &blockDim) {

		// 1. we receive the block and grid size
		__ea dim3 *gridDim_ptr = (__ea dim3*) spu_read_in_mbox();
		__ea dim3 *blockDim_ptr = (__ea dim3*) spu_read_in_mbox();

		gridDim  = *gridDim_ptr;
		blockDim = *blockDim_ptr;

		real_call_kernel<arity> (kernel_ptr);
	}

	template <typename F_, int arity>
	void real_call_kernel(F_, __ea dim3*, __ea dim3*);

	template <typename F_>
	void real_call_kernel<1>(F_ kernel_ptr, __ea dim3 *gridDim, __ea dim3 *blockDim) {
		typedef typename boost::remove_pointer<F_>::type F;

		// 2. we receive all kernel parameters
		typedef typename boost::function_traits <typename F> :: arg1_type ARG1;
		ARG1 *arg1 = (ARG1*) spu_read_in_mbox();

		// 3. which blocks should we calculate
		const unsigned int start_calc = spu_read_in_mbox();
		const unsigned int end_calc = spu_read_in_mbox();

		dim3 blockIdx(start_calc);
		for (; blockIdx.x < end_calc; ++blockIdx.x) {
			for (blockIdx.y=0; blockIdx.y<gridDim->y; ++blockIdx.y) {
				for (blockIdx.z=0; blockIdx.z<gridDim->z; ++blockIdx.z) {

					kernel_ptr(arg1);

				}
			}
		}
	}



		


} // cell

} // cupp

#endif // CUPP_CELL_spe_kernel_call_H

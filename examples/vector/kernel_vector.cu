/*
 * Author:  Jens Breitbart, http://www.gpuified.de/contact/
 *
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#include "deviceT/vector.h"

__global__ void global_function (cupp::deviceT::vector<int> &i) {
	i[threadIdx.x] *= 2;
}

typedef void(*kernelT)(cupp::deviceT::vector<int> &);

kernelT get_kernel() {
	return global_function;
}

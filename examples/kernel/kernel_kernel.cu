/*
 * Author:  Jens Breitbart, http://www.gpuified.de/contact/
 *
 * Copyright: See COPYING file that comes with this distribution
 *
 */

__global__ void global_function (const int i, int &j) {
	j = i;
}

typedef void(*kernelT)(const int, int&);

kernelT get_kernel() {
	return global_function;
}

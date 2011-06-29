/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#include <cstdlib>
#include <iostream>

#include "cupp/device.h"
#include "cupp/kernel.h"

#include "kernel_t.h"

using namespace std;
using namespace cupp;


int main() {
	// lets get a simple CUDA device up and running
	device d;
	
	dim3 block_dim (1);
	dim3 grid_dim  (1);

	test a;
	// generate the kernel
	kernel k (get_kernel(), grid_dim, block_dim);
	
	// call the kernel
	k(d, a);
	
	// NDT
	return EXIT_SUCCESS;
}

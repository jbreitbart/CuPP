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

	int i = 42;
	int j = 23;

	cout << "before kernel call: (" << i << ", " << j << ")" << endl;
	
	dim3 block_dim (1);
	dim3 grid_dim  (1);

	// generate the kernel
	kernel k (get_kernel(), grid_dim, block_dim);
	
	// call the kernel
	k(d, i, j);
	
	cout << "after kernel call: (" << i << ", " << j << ")" << endl;
	
	// NDT
	return EXIT_SUCCESS;
}

/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */
#include <iostream>


#include "cupp/cell/cuda_stub.h"
#include "cupp/device.h"
#include "cupp/kernel.h"

#include "kernel_t.h"

extern spe_program_handle_t kernel;


int main (unsigned long long /*id*/) {
	cupp::device d;

	d.set_spes(1);

	int *mem = new int[10*10];

// 	std::cout << mem << std::endl;

	for (int i=0; i<10*10; ++i) {
		mem[i] = i;
	}

	dim3 block_dim(10);
	dim3 grid_dim(10);

	cupp::kernel k(f, kernel, grid_dim, block_dim);

	k (d, mem);

	for (int i=0; i<10*10; ++i) {
// 		std::cout << mem[i] << ", ";
		if (mem[i] != i*2) {
			std::cerr << std::endl << "Error, somthing is wrong with the kernel" << std::endl;
			break;
		}
	}

	std::cout << std::endl;

	delete[] mem;

	return 0;
}

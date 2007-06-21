#include <cstdlib>
#include <iostream>

#include "device.h"
#include "memory1d.h"
#include "kernel.h"

using namespace std;
using namespace cupp;

typedef void(*kernelT)(cupp::memory1d<int>);

kernelT get_kernel();

int main(int argc, char *argv[]) {
	// lets get a simple CUDA device up and running
	device d;

	// some boring CPU code
	int eight[8];
	for (int i=0; i<8; ++i) {
		eight[i]=i;
	}

	// show me what I have just filled into the array
	cout << "before the kernel call:" << endl;
	for (int i=0; i<8; ++i) {
		cout << eight[i] << ", ";
	}
	cout <<  endl;
	
	// get some memory on the device and fill it with the data from eight
	memory1d<int> mem(d, eight, 8);
	
	dim3 block_dim (8);
	dim3 grid_dim  (1);

	// generate the kernel
	kernel<cupp::memory1d<int> > k (get_kernel(), grid_dim, block_dim);
	
	// call the kernel
	k(d, mem);

	mem.copy_to_host(eight);
	
	// show me what I have just filled into the array
	cout << "after the kernel call:" << endl;
	for (int i=0; i<8; ++i) {
		cout << eight[i] << ", ";
	}
	cout <<  endl;
	
	// NDT
	return EXIT_SUCCESS;
}

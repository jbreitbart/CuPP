/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#include <cstdlib>
#include <iostream>

#include "cupp/device.h"
#include "cupp/vector.h"
#include "cupp/kernel.h"

#include "kernel_t.h"

using namespace std;
using namespace cupp;


int main() {
	// lets get a simple CUDA device up and running
	device d;

	// some boring CPU code
	cupp::vector< cupp::vector<int> > eight (8);
	for (int i=0; i<8; ++i) {
		cupp::vector<int>& t = eight.at(i).get();
		t.push_back(i);
	}

	// show me what I have just filled into the array
	cout << "before the kernel call:" << endl;
	for (int i=0; i<8; ++i) {
		cout << eight.at(i).get().at(0) << ", ";
	}
	cout <<  endl;
	
	dim3 block_dim (8);
	dim3 grid_dim  (1);

	// generate the kernel
	kernel k (get_kernel(), grid_dim, block_dim );
	
	// call the kernel
	k (d, eight);

	// print the array with the new value
	cout << "after the kernel call:" << endl;
	for (int i=0; i<8; ++i) {
		cout << eight.at(i).get().at(0) << ", ";
	}
	cout << endl;
	
	// NDT
	return EXIT_SUCCESS;
}

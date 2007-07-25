#include <cstdlib>
#include <iostream>

#include "device.h"
#include "vector.h"
#include "kernel.h"

using namespace std;
using namespace cupp;

typedef void(*kernelT)(cupp::deviceT::vector<int> &);

kernelT get_kernel();

int main() {
	// lets get a simple CUDA device up and running
	device d;

	// some boring CPU code
	cupp::vector<int> eight;
	for (int i=0; i<8; ++i) {
		eight.push_back(i);
	}

	// show me what I have just filled into the array
	cout << "before the kernel call:" << endl;
	for (int i=0; i<8; ++i) {
		cout << eight[i] << ", ";
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
	for (cupp::vector<int>::iterator i=eight.begin(); i!=eight.end(); ++i) {
		cout << *i << ", ";
		*i = *i * 2;
	}
	cout << endl;

	cout << "multiply every value with 2:" << endl;
	for (cupp::vector<int>::iterator i=eight.begin(); i!=eight.end(); ++i) {
		cout << *i << ", ";
	}
	cout << endl;
	
	// call the kernel
	k (d, eight);

	// show me what I have just filled into the array
	cout << "after the next kernel call:" << endl;
	for (cupp::vector<int>::iterator i=eight.begin(); i!=eight.end(); ++i) {
		cout << *i << ", ";
		*i = *i * 2;
	}
	cout << endl;
	
	// NDT
	return EXIT_SUCCESS;
}

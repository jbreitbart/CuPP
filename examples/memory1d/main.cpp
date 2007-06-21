#include <cstdlib>
#include <iostream>

#include "device.h"
#include "memory1d.h"

using namespace std;
using namespace cupp;

void kernel(memory1d<int> &p);

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

	// call a kernel
	kernel (mem);

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

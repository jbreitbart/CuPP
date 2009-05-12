/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */
#include <iostream>
#include <cstdlib>

#include <sys/time.h>

#include "cupp/cell/cuda_stub.h"
#include "cupp/device.h"
#include "cupp/kernel.h"

#include "kernel_t.h"

extern spe_program_handle_t kernel;

double timediff(timeval tv2, timeval tv1) {
	return (double) (tv2.tv_sec - tv1.tv_sec) + ((double) (tv2.tv_usec - tv1.tv_usec) / 1000000.0);
}


template <typename T>
void fill (T* mem, const int size) {
	for(int i=0; i<size; i++)  {
		mem[i] = (rand() & 0xFF) / static_cast<T>(RAND_MAX);
	}
}

template<>
void fill<int> (int* mem, const int size) {
	for(int i=0; i<size; i++)  {
		mem[i] = static_cast<int>(rand() & 0xFF);
	}
}

template<class T>
T reduce(T *data, int size) {
	T sum = data[0];
	for (int i = 1; i < size; i++) {
		sum += data[i];
	}
	return sum;
}


int main (unsigned long long /*id*/) {
	timeval tv1, tv2;

	cupp::device d;

	int size = 1<<25;    // number of elements to reduce
	int threads = 128;  // number of threads per block

	datatype *mem = new datatype[size];
	datatype *copy = new datatype[size];

	fill (mem, size);
	

	for(int i=0; i<size; i++)  {
		copy[i] = mem[i];
	}

	gettimeofday(&tv1, NULL);
	const datatype cpu_reduced = reduce(copy, size);
	gettimeofday(&tv2, NULL);
	std::cout << "CPU: " << cpu_reduced << " - " << timediff(tv2, tv1) << std::endl;

	dim3 block_dim(threads);
	dim3 grid_dim(size/threads);

	cupp::kernel k(f, &kernel, grid_dim, block_dim);

	for (int i=1; i<=6; ++i) {
		datatype device_reduced = 0;

		d.set_spes(i);

		gettimeofday(&tv1, NULL);
		k (d, mem, &device_reduced);
		gettimeofday(&tv2, NULL);

		if (cpu_reduced != device_reduced) {
			std::cerr << "Error, somthing is wrong with the kernel" << std::endl;
		}


		std::cout << "DEVICE with " << i << " SPEs: " << device_reduced << " - " << timediff(tv2, tv1) << std::endl;
	}

	delete[] mem;
	delete[] copy;

	return 0;
}

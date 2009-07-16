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

typedef union {
  unsigned long long ull;
  unsigned int ui[2];
} addr64;


#include "kernel_t.h"

extern spe_program_handle_t kernel;

double timediff(timeval tv2, timeval tv1) {
	return (double) (tv2.tv_sec - tv1.tv_sec) + ((double) (tv2.tv_usec - tv1.tv_usec) / 1000000.0);
}


template <typename T>
void fill (T* mem, const int size) {
	for(int i=0; i<size; i++)  {
		mem[i] = (rand() & 0xFF) / static_cast<T>(RAND_MAX) + 1;
	}
}

template<>
void fill<int> (int* mem, const int size) {
	for(int i=0; i<size; i++)  {
		mem[i] = static_cast<int>(rand() & 0xFF) + 1;
	}
}


void matrix_mul (const datatype *A, const datatype *B, datatype *C, const int size) {
	for (int i=0; i<size; ++i) {
		for (int j=0; j<size; ++j) {
			for (int k=0; k<size; ++k) {
				C[i*size + j] += A[i*size + k] * B[k*size + j];
			}
		}
	}
}

void print_matrix (const datatype *A, const int size) {
	for (int i=0; i<size; ++i) {
		std::cout << i << ": ";
		for (int j=0; j<size; ++j) {
			std::cout << A[i*size + j] << " ";
		}
		std::cout << std::endl;
	}
}


void *malloc_aligned(size_t size) {
	char* ptr = (char*)malloc(size + 128-1);
	if (!ptr) return NULL;

	const int adr = (int)ptr;

	if ( adr % 128 != 0) {
		ptr += 128 - (adr%128);
	}

	return ptr;
}

int main (unsigned long long /*id*/) {
	timeval tv1, tv2;

	cupp::device d;

	int size = 1<<10;    // number of elements to reduce
	int threads = 16;  // number of threads per block

	std::cout << "size: " << size << std::endl;


	datatype *A = (datatype*) malloc_aligned ( sizeof(datatype)*size*size );
	datatype *B = (datatype*) malloc_aligned ( sizeof(datatype)*size*size );
	datatype *C = (datatype*) malloc_aligned ( sizeof(datatype)*size*size );
	datatype *C_d = (datatype*) malloc_aligned ( sizeof(datatype)*size*size );

	fill (A, size*size);
	fill (B, size*size);

// 	print_matrix (A, size);
// 	std::cout << "=========================" << std::endl;
// 	print_matrix (B, size);


	
// 	gettimeofday(&tv1, NULL);
// 	for (int i=0; i<size*size; ++i) {
// 		C[i] = 0;
// 	}
// 	matrix_mul (A, B, C, size);
// 	gettimeofday(&tv2, NULL);

// 	std::cout << "CPU: " << timediff(tv2, tv1) << std::endl;

// 	print_matrix (C, size);

	dim3 block_dim(threads, threads);
	dim3 grid_dim(size/threads, size/threads);

	cupp::kernel k(f, &kernel, grid_dim, block_dim);

	for (int i=1; i<=6; ++i) {

		d.set_spes(i);
		addr64 a, b;
		a.ull = (unsigned int)A;
		b.ull = (unsigned int)B;

// 		std::cout << A << " - " << B << std::endl;
// 		std::cout << (unsigned int)A << " - " << (unsigned int)B << std::endl;
// 		std::cout << a.ull << " - " << b.ull << std::endl;

		gettimeofday(&tv1, NULL);
		k (d, a, b, C_d, size);
		gettimeofday(&tv2, NULL);

		std::cout << "DEVICE with " << i << " SPEs: " << timediff(tv2, tv1) << std::endl;

// 		print_matrix (C_d, size);
/*		for (int j=0; j<size*size; ++j) {
			if (C[j] != C_d[j]) {
				std::cerr << "Error, somthing is wrong with the kernel " << j << std::endl;
				return 1;
			}
		}*/
	}

// 	delete[] A;
// 	delete[] B;
// 	delete[] C;
// 	delete[] C_d;

	return 0;
}

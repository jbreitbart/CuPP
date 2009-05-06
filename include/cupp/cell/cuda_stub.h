/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */

#ifndef CUPP_CELL_cuda_stub_H
#define CUPP_CELL_cuda_stub_H


struct dim3 {
	int x;
	int y;
	int z;

#ifdef __cplusplus
	dim3 () : x(1), y(1), z(1) {}
	dim3 (int x_) : x(x_), y(1), z(1) {}
	dim3 (int x_, int y_) : x(x_), y(y_), z(1) {}
	dim3 (int x_, int y_, int z_) : x(x_), y(y_), z(z_) {}
#endif
};






#endif // CUPP_CELL_cuda_stub_H

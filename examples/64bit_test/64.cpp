#include <cstdlib>
#include <iostream>

#include "cuda_runtime.h"

#include "kernel_t.h"

using namespace std;

int main( int, char** ) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount); 
    if (deviceCount == 0) { 
        cerr << "There is no device." << endl;
        exit(EXIT_FAILURE);
    } 
    int dev;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (deviceProp.major >= 1) {
            break;
        }
    }
    if (dev == deviceCount) {
        cerr << "There is no device supporting CUDA." << endl;
        exit(EXIT_FAILURE);
    }
    else {
        cudaSetDevice(dev);
    }

    int i = 42;	
    int *d_jp = 0;
    
    cudaMalloc((void**)&d_jp, sizeof(int));
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    
    cudaMemset( d_jp, 0, sizeof(int)); 
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    
    dim3 block_dim (1);
    dim3 grid_dim  (1);
    cudaConfigureCall(grid_dim, block_dim);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    
    cudaSetupArgument(i, 0);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;    

    cudaSetupArgument(d_jp,sizeof(int) );
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    
    cudaLaunch(get_kernel());
    cout << cudaGetErrorString(cudaGetLastError()) << endl;

    int result;
    cudaMemcpy(&result, d_jp, sizeof(int), cudaMemcpyDeviceToHost);
    cout << cudaGetErrorString(cudaGetLastError()) << endl;
    
    cout <<"result " << result << " (should be 666)" << endl;

    return EXIT_SUCCESS;
}

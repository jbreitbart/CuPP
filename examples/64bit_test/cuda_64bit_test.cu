#include <stdlib.h>
#include <stdio.h>


#include "cuda_runtime.h"


int const expected_kernel_result = 666;


/**
 * CUDA kernel that sets the first memory value @a j os pointing to to @c expected_kernel_result.
 */
__global__ void kernel_function (int i, int* j);


/**
 * Returns the CUDA kernel function casted to a <code>char const*</code>.
 */
char const* get_kernel_function();


/**
 * Sets up a CUDA device if one can be found.
 * 
 * @return @c true if a CUDA device has successfully been set up, @c false otherwise.
 */
bool setup_cuda();


/**
 * Checks if CUDA signals an error and prints the error message.
 *
 * @return CUDA error code
 */
cudaError_t check_cuda_error();


int main( int, char** ) 
{

    if ( ! setup_cuda() ) {
        return EXIT_FAILURE;
    }

    int i = 42;	
    int *d_jp = 0;
    
    cudaMalloc((void**)&d_jp, sizeof(int));
    check_cuda_error();
    
    cudaMemset( d_jp, 0, sizeof(int)); 
    check_cuda_error();
    
    dim3 block_dim (1);
    dim3 grid_dim  (1);
    cudaConfigureCall(grid_dim, block_dim);
    check_cuda_error();
    
    cudaSetupArgument(i, 0);
    check_cuda_error();  

    cudaSetupArgument(d_jp,sizeof(int) );
    check_cuda_error();
    
    cudaLaunch( kernel_function );
    // cudaLaunch(get_kernel_function());
    check_cuda_error();



    int result;
    cudaMemcpy(&result, d_jp, sizeof(int), cudaMemcpyDeviceToHost);
    check_cuda_error();
    
    printf("result %d", result );
    printf( " (should be 666)\n");


    if ( expected_kernel_result != result ) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


__global__ void kernel_function(int i, int* j) {
    *j = expected_result;
}

char const* get_kernel_function() {
    return ((char const*) kernel_function);
}




bool setup_cuda()
{
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount); 
    if (deviceCount == 0) { 
        printf( "There is no device.\n" );
        return false;
    } 
    
    int dev = 0;
    for (dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        if (deviceProp.major >= 1) {
            break;
        }
    }
    
    if (dev == deviceCount) {
        printf( "There is no device supporting CUDA.\n" );
        return false;
    }
    
    
    cudaSetDevice(dev);
    
    return true;
} // bool setup_cuda()



cudaError_t check_cuda_error()
{
    cudaError_t error = cudaGetLastError();
    
    if ( cudaSuccess != error ) {
        printf( "%s \n", cudaGetErrorString( error ) );
    }
    
    return error;
} // bool check_cuda_error()


/**
 * Test program for calling a CUDA kernel using the CUDA runtime API 
 * (@c cudaLaunch, @c cudaSetupArgument, @c cudaConfigureCall).
 *
 * A kernel is called and gets two arguments, The first one is ignored (but us needed to trigger
 * the error we are observing). The second argument is a device pointer to memory allocated on
 * the device.
 * The kernel should set the first memory address pointed to to 666.
 * This is checked by copying the memory from the device to the host after the kernel call.
 *
 * However on our 64bit Open SuSE Linux 10.2 we always get an "unspecified launch failure".
 * If we remove the first kernel function parameter (and the associated cudaSetupArgument calls)
 * there is no problem.
 */

#include <stdlib.h>
#include <stdio.h>


#include "cuda_runtime.h"


#define EXPECTED_KERNEL_RESULT 666;
int const expected_kernel_result = EXPECTED_KERNEL_RESULT;


/**
 * CUDA kernel that sets the first memory value @a j os pointing to to @c expected_kernel_result.
 *
 * @param i Dummy parameter but needed to trigger the problem we are observing.
 * @param j Pointer to device memory which can at least hold one @c int value.
 */
__global__ void kernel_function ( int i, int* j);


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
    
    cudaMalloc( (void**) &d_jp, sizeof( int ) );
    check_cuda_error();
    
    cudaMemset( d_jp, 0, sizeof( int ) ); 
    check_cuda_error();
    
    /* Setup CUDA kernel arguments */
    dim3 block_dim( 1 );
    dim3 grid_dim( 1 );
    
    bool const use_native_cuda_kernel_call = false;
    if ( use_native_cuda_kernel_call ) {
        
        /* CUDA native kernel call. */
        kernel_function<<< grid_dim, block_dim >>>( i, d_jp);
        
    } else {
    
        /* CUDA runtime API kernel call. */
        
        cudaConfigureCall( grid_dim, block_dim );
        check_cuda_error();
        
        int offset = 0;
        cudaSetupArgument( i, offset);
        check_cuda_error();	 
        offset += sizeof( i );
    
        cudaSetupArgument( d_jp, offset );
        check_cuda_error();
        offset += sizeof( d_jp );
        
        cudaLaunch( kernel_function );
        check_cuda_error();

    }

    int result = 0;
    cudaMemcpy( &result, d_jp, sizeof( int ), cudaMemcpyDeviceToHost );
    check_cuda_error();
    
    printf( "result %d", result );
    printf( " (should be 666)\n");


    if ( expected_kernel_result != result ) {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}




__global__ void kernel_function( int i, int* j ) {
    *j = EXPECTED_KERNEL_RESULT;
}


bool setup_cuda()
{
    int device_count = 0;
    cudaGetDeviceCount( &device_count ); 
    if ( device_count == 0) { 
        printf( "There is no device.\n" );
        return false;
    } 
    
    int dev = 0;
    for (dev = 0; dev < device_count; ++dev) {
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, dev);
        if (device_prop.major >= 1) {
            break;
        }
    }
    
    if (dev == device_count) {
        printf( "There is no device supporting CUDA.\n" );
        return false;
    }
    
    
    cudaSetDevice( dev );
    
    return true;
} /* bool setup_cuda() */



cudaError_t check_cuda_error()
{
    cudaError_t error = cudaGetLastError();
    
    if ( cudaSuccess != error ) {
        printf( "%s \n", cudaGetErrorString( error ) );
    }
    
    return error;
} /* bool check_cuda_error() */


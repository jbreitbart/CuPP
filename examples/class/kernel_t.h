/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef kernel_t_H
#define kernel_t_H

#if !defined(__CUDACC__)

#include <cupp/device.h>
#include <cupp/runtime.h>

#endif

class test;

struct test_device {
	typedef test_device   device_type;
	typedef test          host_type;
	
	int *arr;
};


#if !defined(__CUDACC__)

class test {
	int *arr;
	int *arr_device;
	
public:
	typedef test_device   device_type;
	typedef test          host_type;

	test() {
		arr = new int[100];
		arr_device = 0;
	}
	
	test(const test &t) {
		// copy constructor, but we don't want to copy any data
		// just want to make sure arr and arr_device are copied from anoth test object
		arr = new int[100];
		arr_device = 0;
	}
	
	~test() {
		delete[] arr;
		cupp::free (arr_device);
	}
	
	
	device_type transform(const cupp::device &d) {
		arr_device = cupp::malloc<int>(100);
		cupp::copy_host_to_device(arr_device, arr, 100);
		
		test_device a;
		a.arr = arr_device;
		return a;
	}
	
	void dirty (cupp::device_reference<device_type> device_ref) {
		//get changed data back
	}

	cupp::device_reference<device_type> get_device_reference(const cupp::device &d) {
		  return cupp::device_reference < device_type > (d, transform(d) );
	}
	
};
#endif

typedef void(*kernelT)(test_device);

// implemented in the .cu file
kernelT get_kernel();

#endif

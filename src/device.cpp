/*
 * Copyright: See LICENSE file that comes with this distribution
 *
 */


#include "cupp/device.h"


#include "cupp/runtime.h"

#include <string>

namespace cupp {

device::device() {
}

device::~device() {
}
void device::sync() const {
	//cupp::thread_synchronize();
}


}

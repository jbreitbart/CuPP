/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef kernel_t_H
#define kernel_t_H

#include "cupp/deviceT/memory1d.h"

typedef void(*kernelT)(cupp::deviceT::memory1d<int>&);

// implemented in the .cu file
kernelT get_kernel();

#endif

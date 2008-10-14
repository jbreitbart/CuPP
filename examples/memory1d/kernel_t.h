/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef kernel_t_H
#define kernel_t_H

#include "cupp/deviceT/memory1d.h"

// implemented in the .cu file
#ifdef __cplusplus
extern "C"  {
#endif

typedef void(*kernelT)(cupp::deviceT::memory1d<int>&);

kernelT get_kernel();

#ifdef __cplusplus
}
#endif


#endif

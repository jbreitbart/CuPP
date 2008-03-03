/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef kernel_t_H
#define kernel_t_H

#include "cupp/deviceT/vector.h"

typedef void(*kernelT)(cupp::deviceT::vector< cupp::deviceT::vector <int> > &);

// implemented in the .cu file
kernelT get_kernel();

#endif

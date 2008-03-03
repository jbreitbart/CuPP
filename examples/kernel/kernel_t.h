/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef kernel_t_H
#define kernel_t_H

typedef void(*kernelT)(const int, int&);

// implemented in the .cu file
kernelT get_kernel();

#endif

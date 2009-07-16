/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef kernel_t_H
#define kernel_t_H

// typedef union {
//   unsigned long long ull;
//   unsigned int ui[2];
// } addr64;


typedef int datatype;

typedef void(*kernelT)(addr64, addr64, datatype*, const int size);

kernelT f;

#endif

/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef kernel_t_H
#define kernel_t_H

typedef int datatype;

typedef void(*kernelT)(datatype*, datatype*);

kernelT f;

#endif

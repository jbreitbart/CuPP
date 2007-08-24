#ifndef kernel_t_H
#define kernel_t_H

typedef void(*kernelT)(cupp::deviceT::memory1d<int>&);

// implemented in the .cu file
kernelT get_kernel();

#endif

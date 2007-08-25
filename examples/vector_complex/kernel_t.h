#ifndef kernel_t_H
#define kernel_t_H

typedef void(*kernelT)(cupp::deviceT::vector< cupp::deviceT::vector <int> > &);

// implemented in the .cu file
kernelT get_kernel();

#endif

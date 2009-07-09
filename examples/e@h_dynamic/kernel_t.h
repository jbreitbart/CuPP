/*
 * Copyright: See COPYING file that comes with this distribution
 *
 */

#ifndef kernel_t_H
#define kernel_t_H

#ifdef __PPU__
#define __ea                    /* ignored */

typedef void (*kernelT) ( 
	       __ea REAL8 *,
	       struct p_spin ,
	       const __ea MultiSFTVector *,
	       __ea ComputeFBuffer *,
	       const REAL8 
	       );


kernelT f;

#endif

static struct dim3 gridDim;
static struct dim3 blockDim;
static struct dim3 blockIdx;

#define LOCAL(a) #a[threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y]
#define __shared__

#define __syncthreads()	\
	} } } \
	for (threadIdx.y=0; threadIdx.y<blockDim.y; ++threadIdx.y) { \
	for (threadIdx.x=0; threadIdx.x<blockDim.x; ++threadIdx.x) { \
	for (threadIdx.z=0; threadIdx.z<blockDim.z; ++threadIdx.z) {

#define START \
	for (threadIdx.y=0; threadIdx.y<blockDim.y; ++threadIdx.y) { \
	for (threadIdx.x=0; threadIdx.x<blockDim.x; ++threadIdx.x) { \
	for (threadIdx.z=0; threadIdx.z<blockDim.z; ++threadIdx.z) {

#define END } } }


/*---------- local DEFINES ----------*/
#define TRUE (1==1)
#define FALSE (1==0)
#define LD_SMALL4       (2.0e-4)		/**< "small" number for REAL4*/
#define OOTWOPI         (1.0 / LAL_TWOPI)	/**< 1/2pi */
#define TWOPI_FLOAT     6.28318530717958f  	/**< single-precision 2*pi */
#define OOTWOPI_FLOAT   (1.0f / TWOPI_FLOAT)	/**< single-precision 1 / (2pi) */


/*----- Macros ----- */
#define SQ(x) ( (x) * (x) )

#define NUM_FACT 6
static const REAL8 inv_fact[NUM_FACT] = { 1.0, 1.0, (1.0/2.0), (1.0/6.0), (1.0/24.0), (1.0/120.0) };

static const Fcomponents _empty_Fcomponents = {0.0, {0.0, 0.0}, {0.0, 0.0}};

struct p_spin{
	PulsarSpins fkdot;
};


typedef union {
	__ea REAL8* arg;
	char dummy[sizeof(__ea REAL8*)];
} arg1T;

typedef union {
	struct p_spin arg;
	char dummy[sizeof(struct p_spin)];
} arg2T;

typedef union {
	__ea MultiSFTVector* arg;
	char dummy[sizeof(__ea MultiSFTVector*)];
} arg3T;

typedef union {
	__ea ComputeFBuffer* arg;
	char dummy[sizeof(__ea ComputeFBuffer*)];
} arg4T;


typedef union {
	REAL8 arg;
	char dummy[sizeof(REAL8)];
} arg5T;


static int temp_sin_cos_2PI_LUT (REAL4 *sin2pix, REAL4 *cos2pix, REAL8 x);

static void
test_spe_ComputeFStat ( __ea REAL8 *Fstat,                 /**< [out] Fstatistic + Fa, Fb */
	       struct p_spin temp,
	       const __ea MultiSFTVector *multiSFTs,    /**< normalized (by DOUBLE-sided Sn!) data-SFTs of all IFOs */
	       __ea ComputeFBuffer *cfBuffer,            /**< CF-internal buffering structure */
	       const REAL8 deltaF
	       );

#endif

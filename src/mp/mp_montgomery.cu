#include <cuda_runtime.h>
#include "mp/mp_montgomery.h"
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "cuda.h"
#include "build_config.h"
#include "log.h"


#ifdef __cplusplus
extern "C" {
#endif


__host__
mon_info *mon_info_copy_to_dev(mon_info *host_info) {
	mon_info *dev_info;
	cudaMalloc((void **) &dev_info, sizeof(mon_info));
	cudaMemcpy(dev_info, host_info, sizeof(mon_info), cudaMemcpyHostToDevice);

	return dev_info;
}


int mon_info_populate(mp_t n, mon_info *info) {

	mpz_t gmp_mu, gmp_n, gmp_R, gmp_Rinv;
	mpz_t gmp_r, gmp_Rsq;

	mpz_init(gmp_mu);
	mpz_init(gmp_n);
	mpz_init(gmp_R);
	mpz_init(gmp_Rinv);
	mpz_init(gmp_r);
	mpz_init(gmp_Rsq);

	/* set n and convert for GMP */
	mp_copy(info->n, n);
	mp_to_mpz(gmp_n, n);

	/* set r <- 2^(LIMB_BITS) */
	mpz_set_ui(gmp_r, 2);
	mpz_pow_ui(gmp_r, gmp_r, LIMB_BITS);

	/* set R <- r^(LIMBS) */
	mpz_pow_ui(gmp_R, gmp_r, LIMBS);
	/* Calculate R^2 mod n */
	/* set R2 <- R^(2) */
	mpz_pow_ui(gmp_Rsq, gmp_R, 2);
	mpz_mod(gmp_Rsq, gmp_Rsq, gmp_n);
	mpz_to_mp(info->R2, gmp_Rsq);
	mpz_clear(gmp_Rsq);


	/* Calculate mu = -n_0^(-1) mod r */
	mpz_t gmp_n0;
	mpz_init_set_ui(gmp_n0, n[0]);
	mpz_invert(gmp_mu, gmp_n0, gmp_r);
	mpz_mul_si(gmp_mu, gmp_mu, -1);
	mpz_mod(gmp_mu, gmp_mu, gmp_r);
	info->mu = mpz_get_ui(gmp_mu) & (LIMB_MASK);
	mpz_clear(gmp_n0);
	mpz_clear(gmp_mu);
	mpz_clear(gmp_r);


	mpz_clear(gmp_Rinv);
	mpz_clear(gmp_R);
	mpz_clear(gmp_n);

	return 0;
}

__device__ __host__
int to_mon(mp_t r, const mp_t a, const mon_info *info) {
	return mon_prod(r, a, info->R2, info);
}

int from_mon(mp_t r, const mp_t a, const mon_info *info) {
	mp_t one;
	mp_set_ui(one, 1);
	return mon_prod(r, a, one, info);
}

__device__ __host__
void mp_div2(mp_t a) {
	for (size_t i = 0; i < LIMBS - 1; i++) {
#if defined(__CUDA_ARCH__) && (LIMB_BITS == 32)
		__shf_r_clamp(a[i], a[i], a[i+1], 1);
#else
		a[i] = (a[i + 1] << (LIMB_BITS - 1)) | (a[i] >> 1);
#endif
	}
	a[LIMBS - 1] >>= 1;
}

__device__ __host__
void mp_shiftl(mp_t a, uint32_t bits, size_t limbs) {
	for (size_t i = limbs - 1; i > 0; i--) {
#if defined(__CUDA_ARCH__) && (LIMB_BITS == 32)
		__shf_l_clamp(a[i], a[i-1], a[i], bits);
#else
		a[i] = (a[i] << bits) | (a[i - 1] >> (LIMB_BITS - bits));
#endif
	}
	a[0] <<= bits;
}

__device__ __host__
void mp_cond_addn_div2(mp_t a, const mp_t n) {
	if (mp_iseven(a)) {
		mp_div2(a);
	} else {
		mp_limb carry = mp_add(a, a, n);
		mp_div2(a);
		if (carry > 0) mp_set_bit(a, BITWIDTH - 1);
	}
}

__device__ __host__
int mon_inv(mp_t r, const mp_t a, const mon_info *info) {
	mp_t u, v, s;
	mp_copy(u, info->n);
	mp_copy(v, a);
	mp_set_ui(r, 0);
	mp_copy(s, info->R2);
	while (mp_cmp_ui(v, 0) > 0) {
		if (mp_iseven(u)) {
			mp_cond_addn_div2(r, info->n);
			mp_div2(u);
		} else if (mp_iseven(v)) {
			mp_cond_addn_div2(s, info->n);
			mp_div2(v);
		} else if (mp_cmp(u, v) > 0) {
			mp_sub_mod(u, u, v, info->n);
			mp_div2(u);
			mp_sub_mod(r, r, s, info->n);
			mp_cond_addn_div2(r, info->n);
		} else {
			mp_sub(v, v, u);
			mp_div2(v);
			mp_sub_mod(s, s, r, info->n);
			mp_cond_addn_div2(s, info->n);
		}
	}
	return 0;
}

__device__ __host__
int mon_prod_cios_cpu(mp_t r, const mp_t a, const mp_t b, const mon_info *info) {
	mp_limb res[LIMBS + 2];
#pragma unroll
	for (int i = 0; i < LIMBS + 2; i++) {
		res[i] = 0;
	}
	mp_limb carry;
	mp_limb tmp_carry;
	mp_limb limb_hi, limb_low;
#pragma unroll
	for (size_t i = 0; i < LIMBS; i++) {

		/* Multiplication */
		carry = 0;
#pragma unroll
		for (size_t j = 0; j < LIMBS; j++) {
			limb_hi = mp_mul_limb(&limb_low, a[j], b[i]);
			limb_hi += mp_limb_addc(res, limb_low, j);
			tmp_carry = carry + limb_hi;
			carry = mp_limb_addc(res, tmp_carry, j + 1) + (tmp_carry < carry);
		}
		res[LIMBS + 1] += carry;

		/* Reduction */
		mp_limb q = info->mu * res[0];
		carry = 0;
#pragma unroll
		for (size_t j = 0; j < LIMBS; j++) {
			limb_hi = mp_mul_limb(&limb_low, info->n[j], q);
			limb_hi += mp_limb_addc(res, limb_low, j);
			tmp_carry = carry + limb_hi;
			carry = mp_limb_addc(res, tmp_carry, j + 1) + (tmp_carry < carry);
		}
		res[LIMBS + 1] += carry;

#pragma unroll
		for (size_t j = 0; j < LIMBS + 1; j++) {
			res[j] = res[j + 1];
		}
		res[LIMBS + 1] = 0;
	}
	if ((res[LIMBS + 1] != 0) || res[LIMBS] != 0 || mp_gt(res, info->n)) {
		mp_limb tmp;

		mp_limb carry = 0;
		for (size_t i = 0; i < LIMBS; i++) {
			tmp = res[i] - info->n[i] - carry;
			carry = (tmp > res[i]);
			r[i] = tmp;
		}
	} else {
		mp_copy(r, res);
	}

	return 0;
}

#ifdef MON_PROD_CIOS
__device__ __host__
int mon_prod_distinct(mp_t r, const mp_t a, const mp_t b, const mon_info *info) {
#ifdef __CUDA_ARCH__
	mp_limb r1 = 0, r2 = 0;

#pragma unroll
	for (int i = 0; i < LIMBS; i++){
		r[i] = 0;
	}

#pragma unroll
	for (size_t i = 0; i < LIMBS; i++) {

		/* Multiplication */
	  	__mad_lo_cc(r[0], a[0], b[i], r[0]);
		for (size_t j = 1; j < LIMBS; j++) {
			__madc_lo_cc(r[j], a[j], b[i], r[j]);
		}
		__addc_cc(r1, r1, (mp_limb)0);
		__addc(r2, r2, (mp_limb)0);
		__mad_hi_cc(r[0+1], a[0], b[i], r[0+1]);

		for (size_t j = 1; j < LIMBS-1; j++) {
			__madc_hi_cc(r[j+1], a[j], b[i], r[j+1]);
		}
		__madc_hi_cc(r1, a[LIMBS-1], b[i], r1);
		__addc(r2, r2, (mp_limb)0);

		/* Reduction */
		mp_limb q = info->mu * r[0];

	  	__mad_lo_cc(r[0], info->n[0], q, r[0]);
		for (size_t j = 1; j < LIMBS; j++) {
			__madc_lo_cc(r[j], info->n[j], q, r[j]);
		}
		__addc_cc(r1, r1, (mp_limb)0);
		__addc(r2, r2, (mp_limb)0);

	  	__mad_hi_cc(r[0], info->n[0], q, r[0+1]);
		for (size_t j = 1; j < LIMBS-1; j++) {
			__madc_hi_cc(r[j], info->n[j], q, r[j+1]);
		}
		__madc_hi_cc(r[LIMBS-1], info->n[LIMBS-1], q, r1);
		__addc(r1, r2, (mp_limb)0);
		r2 = 0;
	}
	if(r1 | mp_gt(r, info->n)){
		__sub_cc(r[0], r[0], info->n[0]);
#pragma unroll
		for (size_t i = 1; i < LIMBS; i++) {
			__subc_cc(r[i], r[i], info->n[i]);
		}
	}
	return 0;

#else /* CPU implementation */
  return mon_prod(r, a, b, info);
#endif
}


__device__ __host__
int mon_square(mp_t r, const mp_t a, const mon_info *info) {
#ifdef __CUDA_ARCH__

	mp_limb res[LIMBS + 2];

#pragma unroll
	for (int i = 0; i < LIMBS + 2; i++){
		res[i] = 0;
	}

#pragma unroll
	for (size_t i = 0; i < LIMBS; i++) {

	if(i > 0){
		mp_limb tmp[LIMBS + 2] = {0};

		/* Multiplication of i != j*/
		// Low
		__mad_lo_cc(tmp[0], a[0], a[i], tmp[0]);
		for (size_t j = 1; j < i; j++) {
			__madc_lo_cc(tmp[j], a[j], a[i], tmp[j]);
	  	}
	  	__addc(tmp[i], (mp_limb)0, (mp_limb)0);

		// High
		__mad_hi_cc(tmp[0+1], a[0], a[i], tmp[0+1]);
		for (size_t j = 1; j < i; j++) {
			__madc_hi_cc(tmp[j+1], a[j], a[i], tmp[j+1]);
		}
	  	__addc(tmp[i+1], (mp_limb)0, (mp_limb)0);

	  	// Double and add to res
		__add_cc(tmp[0], tmp[0], tmp[0]);
		for (size_t j = 1; j < i+2; j++) {
			__addc_cc(tmp[j], tmp[j], tmp[j]);
		}
		__add_cc(res[0], res[0], tmp[0]);
#pragma unroll
		for (size_t j = 1; j < LIMBS+2; j++) {
			__addc_cc(res[j], res[j], tmp[j]);
		}
	}


	/* Multiplication of i == j */
	__mad_lo_cc(res[i], a[i], a[i], res[i]);
	__madc_hi_cc(res[i+1], a[i], a[i], res[i+1]);
	for(int j = i+1; j < LIMBS; j++){
		__addc_cc(res[j+1], res[j+1], (mp_limb)0);
	}
	__addc_cc(res[LIMBS], res[LIMBS], (mp_limb)0);
	__addc(res[LIMBS+1], res[LIMBS+1], (mp_limb)0);

	/* Reduction */
	mp_limb q = info->mu * res[0];

	__mad_lo_cc(res[0], info->n[0], q, res[0]);
#pragma unroll
	for (size_t j = 1; j < LIMBS; j++) {
		__madc_lo_cc(res[j], info->n[j], q, res[j]);
	}
	__addc_cc(res[LIMBS], res[LIMBS], (mp_limb)0);
	__addc(res[LIMBS+1], res[LIMBS+1], (mp_limb)0);

	__mad_hi_cc(res[0], info->n[0], q, res[0+1]);
#pragma unroll
	for (size_t j = 1; j < LIMBS; j++) {
		__madc_hi_cc(res[j], info->n[j], q, res[j+1]);
	}
	__addc(res[LIMBS], res[LIMBS+1], (mp_limb)0);
		res[LIMBS+1] = 0;
	}

	if(res[LIMBS] | mp_gt(res, info->n)){
		__sub_cc(r[0], res[0], info->n[0]);
#pragma unroll
		for (size_t i = 1; i < LIMBS; i++) {
			__subc_cc(r[i], res[i], info->n[i]);
		}
	} else {
		mp_copy(r, res);
	}

	return 0;

#else /* CPU */
  return mon_prod(r, a, a, info);
#endif

}

__device__ __host__
int mon_prod(mp_t r, const mp_t a, const mp_t b, const mon_info *info) {
#ifdef __CUDA_ARCH__
	mp_limb res[LIMBS + 2];

#pragma unroll
	for (int i = 0; i < LIMBS + 2; i++){
		res[i] = 0;
	}

#pragma unroll
	for (size_t i = 0; i < LIMBS; i++) {

		/* Multiplication */
	  __mad_lo_cc(res[0], a[0], b[i], res[0]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			__madc_lo_cc(res[j], a[j], b[i], res[j]);
		}
		__addc_cc(res[LIMBS], res[LIMBS], (mp_limb)0);
		__addc(res[LIMBS+1], res[LIMBS+1], (mp_limb)0);

	  __mad_hi_cc(res[0+1], a[0], b[i], res[0+1]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			__madc_hi_cc(res[j+1], a[j], b[i], res[j+1]);
		}
		__addc(res[LIMBS+1], res[LIMBS+1], (mp_limb)0);

		/* Reduction */
		mp_limb q = info->mu * res[0];

	  	__mad_lo_cc(res[0], info->n[0], q, res[0]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			__madc_lo_cc(res[j], info->n[j], q, res[j]);
		}
		__addc_cc(res[LIMBS], res[LIMBS], (mp_limb)0);
		__addc(res[LIMBS+1], res[LIMBS+1], (mp_limb)0);

	  	__mad_hi_cc(res[0], info->n[0], q, res[0+1]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			__madc_hi_cc(res[j], info->n[j], q, res[j+1]);
		}
		__addc(res[LIMBS], res[LIMBS+1], (mp_limb)0);
		res[LIMBS+1] = 0;

	}
	if(res[LIMBS] | mp_gt(res, info->n)){
		__sub_cc(r[0], res[0], info->n[0]);
#pragma unroll
		for (size_t i = 1; i < LIMBS; i++) {
			__subc_cc(r[i], res[i], info->n[i]);
		}
	} else {
		mp_copy(r, res);
	}

	return 0;

#else /* CPU implementation */
	return mon_prod_cios_cpu(r, a, b, info);
#endif

}

#endif


#ifdef MON_PROD_FIPS

__device__ __host__
int mon_square(mp_t r, const mp_t a, const mon_info *info) {
	return mon_prod(r, a, a, info);
}

__device__ __host__
int mon_prod_distinct(mp_t r, const mp_t a, const mp_t b, const mon_info *info) {
	// TODO: Implement FIPS for r != a & r != b
	return mon_prod(r, a, b, info);
}

__device__ __host__
int mon_prod(mp_t r, const mp_t a, const mp_t b, const mon_info *info) {
#ifdef __CUDA_ARCH__
	mp_limb res[LIMBS + 2];
#pragma unroll
	for (int i = 0; i < LIMBS + 2; i++){
		res[i] = 0;
	}
	mp_limb t = 0, u = 0, v = 0;
	for (size_t i = 0; i < LIMBS; i++) {
		for (size_t j = 0; j < i; j++) {
			__mad_lo_cc(v, a[j], b[i-j], v);
			__madc_hi_cc(u, a[j], b[i-j], u);
			__addc(t, t, (mp_limb)0);

			__mad_lo_cc(v, res[j], info->n[i-j], v);
			__madc_hi_cc(u, res[j], info->n[i-j], u);
			__addc(t, t, (mp_limb)0);
		}

		__mad_lo_cc(v, a[i], b[0], v);
		__madc_hi_cc(u, a[i], b[0], u);
		__addc(t, t, (mp_limb)0);

		res[i] = info->mu * v;

		__mad_lo_cc(v, res[i], info->n[0], v);
		__madc_hi_cc(u, res[i], info->n[0], u);
		__addc(t, t, (mp_limb)0);
		v = u;
		u = t;
		t = 0;
  	}
#pragma unroll
	for (size_t i = LIMBS; i < 2*LIMBS; i++) {
		for (size_t j = i-LIMBS+1; j < LIMBS; j++) {
			__mad_lo_cc(v, a[j], b[i-j], v);
			__madc_hi_cc(u, a[j], b[i-j], u);
			__addc(t, t, (mp_limb)0);

			__mad_lo_cc(v, res[j], info->n[i-j], v);
			__madc_hi_cc(u, res[j], info->n[i-j], u);
			__addc(t, t, (mp_limb)0);
		}
		res[i-LIMBS] = v;
		v = u;
		u = t;
		t = 0;
  	}
  res[LIMBS] = v;

  /* Conditional subtraction */
	if((res[LIMBS+1] != 0) || res[LIMBS] != 0 || mp_gt(res, info->n)){
		__sub_cc(r[0], res[0], info->n[0]);
#pragma unroll
		for (size_t i = 1; i < LIMBS; i++) {
			__subc_cc(r[i], res[i], info->n[i]);
		}
	} else {
		mp_copy(r, res);
	}

	return 0;

#else /* CPU Implementation (CIOS) */
	return mon_prod_cios_cpu(r, a, b, info);
#endif

}

#endif


#ifdef MON_PROD_FIOS
__device__ __host__
int mon_square(mp_t r, const mp_t a, const mon_info *info) {
	return mon_prod(r, a, a, info);
}

__device__ __host__
int mon_prod_distinct(mp_t r, const mp_t a, const mp_t b, const mon_info *info) {
  return mon_prod(r, a, b, info);
}

__device__ __host__
int mon_prod(mp_t r, const mp_t a, const mp_t b, const mon_info *info) {
#ifdef __CUDA_ARCH__
	mp_limb res[LIMBS + 2];
#pragma unroll
	for (int i = 0; i < LIMBS + 2; i++){
		res[i] = 0;
	}

	mp_limb c = 0, s = 0, q = 0, t = 0;
	for (size_t i = 0; i < LIMBS; i++) {
		__mad_lo_cc(s, a[0], b[i], res[0]);
		__madc_hi(t, a[0], b[i], 0);

		q = info->mu * s;
		__mad_lo_cc(s, info->n[0], q, s);
		__madc_hi_cc(c, info->n[0], q, t);
		__addc_cc(t, 0, 0);

		for (size_t j = 1; j < LIMBS; j++) {
			__madc_lo_cc(s, a[j], b[i], c);
			__madc_hi(c, a[j], b[i], t);
			
			__mad_lo_cc(s, info->n[j], q, s);
			__madc_hi_cc(c, info->n[j], q, c);
			__addc(t, 0, 0);

			__add_cc(res[j-1], s, res[j]);

		}
		__addc_cc(res[LIMBS-1], res[LIMBS], c);
		__addc(res[LIMBS], res[LIMBS+1], t);

		res[LIMBS+1] = 0;
	}

	/* Conditional subtraction */
	if((res[LIMBS+1] != 0) || res[LIMBS] != 0 || mp_gt(res, info->n)){
		__sub_cc(r[0], res[0], info->n[0]);
#pragma unroll
		for (size_t i = 1; i < LIMBS; i++) {
			__subc_cc(r[i], res[i], info->n[i]);
		}
	} else {
		mp_copy(r, res);
	}

	return 0;

#else /* CPU Implementation (CIOS) */
	return mon_prod_cios_cpu(r, a, b, info);
#endif
}

#endif


#ifdef MON_PROD_CIOS_XMAD
__device__ __host__
int mon_prod_distinct(mp_t r, const mp_t a, const mp_t b, const mon_info *info) {
  return mon_prod(r, a, b, info);
}


__device__ __host__
int mon_square(mp_t r, const mp_t a, const mon_info *info) {
  return mon_prod(r, a, a, info);
}



#define xmadll_cc(r, a, b, c)                                    \
	asm volatile ("{ 																	       \n\t"         \
								".reg .u16			%alo, %ahi, %blo, %bhi;    \n\t"  \
								".reg .u32			%t;                        \n\t"  \
								"mov.b32				{%alo, %ahi}, %1;					 \n\t"  \
								"mov.b32				{%blo, %bhi}, %2;					 \n\t"  \
								"mul.wide.u16		%t, %alo, %blo;						 \n\t"  \
								"add.cc.u32		%0, %3, %t;								   \n\t"  \
								"}"	: "=r"(r) : "r" (a), "r" (b), "r" (c));
#define xmadll_c_cc(r, a, b, c)                                    \
	asm volatile ("{ 																	       \n\t"         \
								".reg .u16			%alo, %ahi, %blo, %bhi;    \n\t"  \
								".reg .u32			%t;                        \n\t"  \
								"mov.b32				{%alo, %ahi}, %1;					 \n\t"  \
								"mov.b32				{%blo, %bhi}, %2;					 \n\t"  \
								"mul.wide.u16		%t, %alo, %blo;						 \n\t"  \
								"addc.cc.u32		%0, %3, %t;								 \n\t"  \
								"}"	: "=r"(r) : "r" (a), "r" (b), "r" (c));

#define xmadhl_cc(r, a, b, c)                                    \
	asm volatile ("{ 																	       \n\t"         \
								".reg .u16			%alo, %ahi, %blo, %bhi;    \n\t"  \
								".reg .u32			%t;                        \n\t"  \
								"mov.b32				{%alo, %ahi}, %1;					 \n\t"  \
								"mov.b32				{%blo, %bhi}, %2;					 \n\t"  \
								"mul.wide.u16		%t, %ahi, %blo;						 \n\t"  \
								"add.cc.u32		%0, %3, %t;								   \n\t"  \
								"}"	: "=r"(r) : "r" (a), "r" (b), "r" (c));
#define xmadhl_c_cc(r, a, b, c)                                    \
	asm volatile ("{ 																	       \n\t"   \
								".reg .u16			%alo, %ahi, %blo, %bhi;    \n\t"  \
								".reg .u32			%t;                        \n\t"  \
								"mov.b32				{%alo, %ahi}, %1;					 \n\t"  \
								"mov.b32				{%blo, %bhi}, %2;					 \n\t"  \
								"mul.wide.u16		%t, %ahi, %blo;						 \n\t"  \
								"addc.cc.u32		%0, %3, %t;								 \n\t"  \
								"}"	: "=r"(r) : "r" (a), "r" (b), "r" (c));

#define xmadlh_cc(r, a, b, c)                                    \
	asm volatile ("{ 																	       \n\t"         \
								".reg .u16			%alo, %ahi, %blo, %bhi;    \n\t"  \
								".reg .u32			%t;                        \n\t"  \
								"mov.b32				{%alo, %ahi}, %1;					 \n\t"  \
								"mov.b32				{%blo, %bhi}, %2;					 \n\t"  \
								"mul.wide.u16		%t, %alo, %bhi;						 \n\t"  \
								"add.cc.u32		%0, %3, %t;								   \n\t"  \
								"}"	: "=r"(r) : "r" (a), "r" (b), "r" (c));
#define xmadlh_c_cc(r, a, b, c)                                    \
	asm volatile ("{ 																	       \n\t"         \
								".reg .u16			%alo, %ahi, %blo, %bhi;    \n\t"  \
								".reg .u32			%t;                        \n\t"  \
								"mov.b32				{%alo, %ahi}, %1;					 \n\t"  \
								"mov.b32				{%blo, %bhi}, %2;					 \n\t"  \
								"mul.wide.u16		%t, %alo, %bhi;						 \n\t"  \
								"addc.cc.u32		%0, %3, %t;								 \n\t"  \
								"}"	: "=r"(r) : "r" (a), "r" (b), "r" (c));

#define xmadhh_cc(r, a, b, c)                                    \
	asm volatile ("{ 																	       \n\t"         \
								".reg .u16			%alo, %ahi, %blo, %bhi;    \n\t"  \
								".reg .u32			%t;                        \n\t"  \
								"mov.b32				{%alo, %ahi}, %1;					 \n\t"  \
								"mov.b32				{%blo, %bhi}, %2;					 \n\t"  \
								"mul.wide.u16		%t, %ahi, %bhi;						 \n\t"  \
								"add.cc.u32		%0, %3, %t;								   \n\t"  \
								"}"	: "=r"(r) : "r" (a), "r" (b), "r" (c));
#define xmadhh_c_cc(r, a, b, c)                                    \
	asm volatile ("{ 																	       \n\t"         \
								".reg .u16			%alo, %ahi, %blo, %bhi;    \n\t"  \
								".reg .u32			%t;                        \n\t"  \
								"mov.b32				{%alo, %ahi}, %1;					 \n\t"  \
								"mov.b32				{%blo, %bhi}, %2;					 \n\t"  \
								"mul.wide.u16		%t, %ahi, %bhi;						 \n\t"  \
								"addc.cc.u32		%0, %3, %t;								 \n\t"  \
								"}"	: "=r"(r) : "r" (a), "r" (b), "r" (c));

__device__ __host__
int mon_prod(mp_t r, const mp_t a, const mp_t b, const mon_info *info) {
#ifdef __CUDA_ARCH__
	mp_limb res[LIMBS + 2];

#pragma unroll
	for (int i = 0; i < LIMBS + 2; i++){
		res[i] = 0;
	}

#pragma unroll
	for (size_t i = 0; i < LIMBS; i++) {
		mp_limb row[LIMBS + 2] = {0};

		xmadlh_cc(row[0], a[0], b[i], row[0]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
		  	xmadlh_c_cc(row[j], a[j], b[i], row[j]);
		}
		xmadhl_cc(row[0], a[0], b[i], row[0]);

#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			xmadhl_c_cc(row[j], a[j], b[i], row[j]);
		}
		__addc(row[LIMBS], row[LIMBS], (mp_limb)0);

		// shift row left by 16b
		mp_shiftl(row, 16, LIMBS+2);

		xmadll_cc(res[0], a[0], b[i], res[0]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			xmadll_c_cc(res[j], a[j], b[i], res[j]);
		}
		__addc_cc(res[LIMBS], res[LIMBS], (mp_limb)0);
		__addc(res[LIMBS+1], res[LIMBS+1], (mp_limb)0);

		xmadhh_cc(res[0+1], a[0], b[i], res[0+1]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			xmadhh_c_cc(res[j+1], a[j], b[i], res[j+1]);
		}
		__addc(res[LIMBS+1], res[LIMBS+1], (mp_limb)0);

		mp_add_limbs(res, res, row, LIMBS+2);

#pragma unroll
		for (size_t j = 0; j < LIMBS+2; j++){
			row[j] = 0;
		}

		/* Reduction */
		mp_limb q = info->mu * res[0];

		xmadlh_cc(row[0], info->n[0], q, row[0]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			xmadlh_c_cc(row[j], info->n[j], q, row[j]);
		}
		xmadhl_cc(row[0], info->n[0], q, row[0]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			xmadhl_c_cc(row[j], info->n[j], q, row[j]);
		}
		__addc(row[LIMBS], row[LIMBS], (mp_limb)0);

		// shift row left by 16b
		mp_shiftl(row, 16, LIMBS+2);

		xmadll_cc(res[0], info->n[0], q, res[0]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			xmadll_c_cc(res[j], info->n[j], q, res[j]);
		}
		__addc_cc(res[LIMBS], res[LIMBS], (mp_limb)0);
		__addc(res[LIMBS+1], res[LIMBS+1], (mp_limb)0);

		xmadhh_cc(res[0+1], info->n[0], q, res[0+1]);
#pragma unroll
		for (size_t j = 1; j < LIMBS; j++) {
			xmadhh_c_cc(res[j+1], info->n[j], q, res[j+1]);
		}
		__addc(res[LIMBS+1], res[LIMBS+1], (mp_limb)0);

		mp_add_limbs(res, res, row, LIMBS+2);

#pragma unroll
		for (size_t j = 0; j < LIMBS+1; j++){
			res[j] = res[j+1];
		}

		res[LIMBS+1] = 0;
	}

	if(res[LIMBS] | mp_gt(res, info->n)){
		__sub_cc(r[0], res[0], info->n[0]);
#pragma unroll
		for (size_t i = 1; i < LIMBS; i++) {
			__subc_cc(r[i], res[i], info->n[i]);
		}
	} else {
		mp_copy(r, res);
	}

	return 0;

#else /* CPU Implementation (CIOS) */
	return mon_prod_cios_cpu(r, a, b, info);
#endif

}

#endif


__host__ __device__
void mp_print_n(const mp_t a, size_t n) {
#pragma unroll
	for (int i = n - 1; i >= 0; i--){
		printf("(%u)[%010" _PRI_ulimb "]", i, a[i]);
	}
	printf("\n");
}


__host__ __device__
void mon_exp(mp_t r, const mp_t b, const mp_t e, const mon_info *info) {
	/* Catch zero exponent */
	if (mp_cmp_ui(e, 0) == 0) {
		mp_set_ui(r, 1);
		return;
	}

	/* Binary left to right */
	int bit = BITWIDTH - 1;
	/* Find MSB == 1; */
	while (!mp_test_bit(e, bit)) {
		bit--;
	}
	mp_copy(r, b);
	bit--;

	while (bit >= 0) {
		mon_prod(r, r, r, info);
		if (mp_test_bit(e, bit)) {
			mon_prod(r, r, b, info);
		}
		bit--;
	}
}

__host__
void print_info(const mon_info *info) {
	printf("MonProd info struct:\n");
	mp_printf("\tn: %Zi\n", info->n);
	mp_printf("\tR2: %Zi\n", info->R2);
	printf("\tmu: %d\n", info->mu);
}

__device__
void print_info_dev(const mon_info *info) {
	printf("MonProd info struct:\n\tn: ");
	mp_print(info->n);
	printf("\n\tr^2: ");
	mp_print(info->R2);
	printf("\n\tmu: %d", info->mu);
}


#ifdef __cplusplus
}
#endif

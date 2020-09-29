#include <inttypes.h>
#include <cuda_runtime.h>
#include "mp/mp.h"

#ifdef __cplusplus
extern "C" {
#endif

__host__ __device__
void mp_print(const mp_t a) {
#pragma unroll
	for (int i = LIMBS - 1; i >= 0; i--)
		printf("(%d)[%010" _PRI_ulimb  "u]", i, a[i]);
	printf("\n");
}

void mp_print_hex(const mp_t a) {
#pragma unroll
	for (int i = LIMBS - 1; i >= 0; i--)
		printf("(%d)[0x%08" _PRI_xlimb "x]", i, a[i]);
	printf("\n");
}

void mp_print_hex_limbs(const mp_t a, size_t limbs) {
#pragma unroll
	for (int i = limbs - 1; i >= 0; i--)
		printf("(%d)[0x%08" _PRI_xlimb "x]", i, a[i]);
	printf("\n");
}

__host__ __device__
void mp_init(mp_t *a) {
	a = (mp_t *) malloc(LIMBS * sizeof(mp_limb));
	mp_set_ui(*a, 0);
}

__host__
void mp_dev_init(mp_p *a) {
	cudaMalloc((void **) a, LIMBS * sizeof(mp_limb));
	cudaMemset(*a, 0, LIMBS * sizeof(mp_limb));
}

__host__
void mp_dev_init_limbs(mp_p *a, size_t limbs) {
	cudaMalloc((void **) a, limbs * sizeof(mp_limb));
	cudaMemset(*a, 0, limbs * sizeof(mp_limb));
}

__host__ __device__
void mp_free(mp_t a) {
	free(a);
}

__host__ __device__
void mp_set_ui(mp_t a, mp_limb s) {
	a[0] = s;
#pragma unroll
	for (size_t i = 1; i < LIMBS; i++)
		a[i] = 0;
}

__host__ __device__
mp_limb mp_limb_addc(mp_t a, const mp_limb s, const size_t limb) {
	mp_limb carry = 0;
#ifdef __CUDA_ARCH__
	__add_cc(a[limb], a[limb], s);
	__addcy(carry);
#else
	a[limb] = a[limb] + s;
	carry = (a[limb] < s);
#endif
	return carry;
}

__host__ __device__
mp_limb mp_limb_subc(mp_t a, const mp_limb s, const size_t limb) {
	mp_limb carry = 0;
#ifdef __CUDA_ARCH__
	__sub_cc(a[limb], a[limb], s);
	__addcy(carry);
#else
	mp_limb tmp = a[limb];
	a[limb] = a[limb] - s;
	carry = (a[limb] > tmp);
#endif
	return carry;
}

__host__ __device__
void mp_copy(mp_t a, const mp_t b) {
#pragma unroll
	for (int i = 0; i < LIMBS; i++) {
		a[i] = b[i];
	}
}

__host__ __device__
void mp_copy_sc(mp_strided_t a, const size_t a_elem, const mp_t b) {
#pragma unroll
	for (int i = 0; i < LIMBS; i++) {
		a[_S_IDX(a_elem, i)] = b[i];
	}
}

__host__ __device__
void mp_copy_cs(mp_t a,
				const mp_strided_t b, const size_t b_elem) {
#pragma unroll
	for (int i = 0; i < LIMBS; i++) {
		a[i] = b[_S_IDX(b_elem, i)];
	}
}

__host__ __device__
void mp_copy_ss(mp_strided_t a, const size_t a_elem,
				const mp_strided_t b, const size_t b_elem) {
#pragma unroll
	for (int i = 0; i < LIMBS; i++) {
		a[_S_IDX(a_elem, i)] = b[_S_IDX(b_elem, i)];
	}
}


__host__
void mp_copy_to_dev(mp_p dev_a, const mp_t b) {
	cudaMemcpy(dev_a, b, LIMBS * sizeof(mp_limb), cudaMemcpyHostToDevice);
}

__host__
void mp_copy_to_dev_limbs(mp_p dev_a, const mp_t b, const size_t limbs) {
	cudaMemcpy(dev_a, b, limbs * sizeof(mp_limb), cudaMemcpyHostToDevice);
}

__host__
void mp_copy_from_dev(mp_t a, const mp_p dev_b) {
	cudaMemcpy(a, dev_b, LIMBS * sizeof(mp_limb), cudaMemcpyDeviceToHost);
}

__host__
mp_limb mp_mul_ui(mp_t r, const mp_t a, const mp_limb s) {
#ifdef __CUDA_ARCH__

	mp_limb carry = 0;
	mp_set_ui(r, 0);
	__mad_lo_cc(r[0], a[0], s, r[0]);
#pragma unroll
	for (size_t j = 1; j < LIMBS; j++) {
		__madc_lo_cc(r[j], a[j], s, r[j]);
		}
		__addc(carry, carry, (mp_limb)0);

	__mad_hi_cc(r[0+1], a[0], s, r[0+1]);
#pragma unroll
	for (size_t j = 1; j < LIMBS-1; j++) {
		__madc_hi_cc(r[j+1], a[j], s, r[j+1]);
	}
		__addc(carry, carry, (mp_limb)0);
	return carry;
#else
	mp_t mptmp;
	mp_set_ui(mptmp, 0);
	mp_limb carry = 0;
	__mp_2limb tmp;

#pragma unroll
	for (size_t i = 0; i < LIMBS - 1; i++) {
		tmp = (s * (__mp_2limb) a[i]) + carry;
		/* Lower bits */
		carry = mp_limb_addc(mptmp, (mp_limb) ((tmp) & LIMB_MASK), i);
		/* Upper bits */
		carry = mp_limb_addc(mptmp, (mp_limb) (((tmp >> LIMB_BITS) + carry) & LIMB_MASK), i + 1);
	}
	tmp = (s * (__mp_2limb) a[LIMBS - 1]) + carry;
	carry = mp_limb_addc(mptmp, (mp_limb) (tmp & LIMB_MASK), LIMBS - 1);
	carry += (tmp >> LIMB_BITS) & LIMB_MASK;

	mp_copy(r, mptmp);
	return carry;
#endif
}

__host__ __device__
mp_limb mp_mul_limb(mp_limb *r, mp_limb a, mp_limb b) {
#ifdef __CUDA_ARCH__
	__mul_lo(*r, a, b);
	mp_limb ret;
	__mul_hi(ret, a, b);
	return ret;
#else
	__mp_2limb tmp = ((__mp_2limb) b * (__mp_2limb) a);
	*r = (tmp & LIMB_MASK);
	return (mp_limb) ((tmp >> LIMB_BITS) & LIMB_MASK);
#endif
}

__host__ __device__
mp_limb mp_add(mp_t r, const mp_t a, const mp_t b) {
	mp_limb carry = 0;
#ifdef __CUDA_ARCH__

	__add_cc(r[0], a[0], b[0]);
#pragma unroll
	for (size_t i = 1; i < LIMBS; i++) {
		__addc_cc(r[i], a[i], b[i]);
	}

	__addcy(carry);
#else
	mp_copy(r, a);
#pragma unroll
	for (size_t i = 0; i < LIMBS; i++) {
		carry = mp_limb_addc(r, b[i] + carry, i);
	}
#endif
	return carry;
}


__host__ __device__
mp_limb mp_add_limbs(mp_t r, const mp_t a, const mp_t b, size_t limbs) {
	mp_limb carry = 0;
#ifdef __CUDA_ARCH__

	__add_cc(r[0], a[0], b[0]);
#pragma unroll
	for (size_t i = 1; i < limbs; i++) {
		__addc_cc(r[i], a[i], b[i]);
	}

	__addcy(carry);
#else
	mp_copy(r, a);
#pragma unroll
	for (size_t i = 0; i < limbs; i++) {
		carry = mp_limb_addc(r, b[i] + carry, i);
	}
#endif
	return carry;
}


__host__ __device__
void mp_add_mod(mp_t r, const mp_t a, const mp_t b, const mp_t n) {
	mp_limb carry;
	carry = mp_add(r, a, b);
	// If result overflowed subtract n
	if (carry > 0 || mp_gt(r, n)) {
		mp_sub(r, r, n);
	}
}


__host__ __device__
mp_limb mp_add_ui(mp_t r, const mp_t a, const mp_limb s) {
	mp_copy(r, a);
	mp_limb carry = mp_limb_addc(r, s, 0);
#pragma unroll
	for (size_t i = 1; i < LIMBS; i++) {
		carry = mp_limb_addc(r, carry, i);
	}
	return carry;
}


__host__ __device__
mp_limb mp_sub(mp_t r, const mp_t a, const mp_t b) {
	mp_limb carry = 0;
#ifdef __CUDA_ARCH__
	__sub_cc(r[0], a[0], b[0]);
#pragma unroll
	for (size_t i = 1; i < LIMBS; i++) {
		__subc_cc(r[i], a[i], b[i]);
	}
	__subc(carry, (mp_limb)0, (mp_limb)0);
#else
	mp_t tmp;
	mp_copy(tmp, a);
#pragma unroll
	for (size_t i = 0; i < LIMBS; i++) {
		carry = mp_limb_subc(tmp, b[i] + carry, i);
		r[i] = tmp[i];
	}
#endif

	return carry;
}


__host__ __device__
void mp_sub_mod(mp_t r, const mp_t a, const mp_t b, const mp_t n) {
	if (mp_sub(r, a, b)) {
		mp_add(r, r, n);
	}
}


__host__ __device__
void mp_sub_ui(mp_t r, const mp_t a, const mp_limb s) {
	mp_t tmp;
	mp_init(&tmp);
	mp_set_ui(tmp, s);
	mp_sub(r, a, tmp);
}

__host__ __device__
void mp_sl_limbs(mp_t a, size_t limbs) {
	if (limbs == 0) return;
#pragma unroll
	for (size_t i = LIMBS - 1; i >= limbs; i--) {
		a[i] = a[i - limbs];
	}
#pragma unroll
	for (size_t i = 0; i < limbs; i++) {
		a[i] = 0;
	}
}


__host__ __device__
void mp_sr_limbs(mp_t a, size_t limbs) {
#pragma unroll
	for (size_t i = 0; i < LIMBS - limbs; i++) {
		a[i] = a[i + limbs];
	}
#pragma unroll
	for (size_t i = LIMBS - 1; i >= LIMBS - limbs; i--) {
		a[i] = 0;
	}
}

__host__ __device__
void mp_mul(mp_t r, const mp_t a, const mp_t b) {
	mp_t tmp;
	mp_t tmp2;
	mp_set_ui(tmp2, 0);

#pragma unroll
	for (size_t i = 0; i < LIMBS; i++) {
		mp_set_ui(tmp, 0);
		mp_mul_ui(tmp, a, b[i]);
		mp_sl_limbs(tmp, i);
		mp_add(tmp2, tmp2, tmp);
	}
	mp_copy(r, tmp2);
}

__host__ __device__
int mp_gt(const mp_t a, const mp_t b) {

#ifdef __CUDA_ARCH__
	mp_limb tmp;
	  __sub_cc(tmp, b[0], a[0]);
#pragma unroll
	  for (size_t i = 1; i < LIMBS; i++) {
		  __subc_cc(tmp, b[i], a[i]);
	  }
	  __subc(tmp, (mp_limb)0, (mp_limb)0);
	  return (tmp > 0) ? 1 : 0;
#else
	mp_t tmp;
	return (mp_sub(tmp, b, a) > 0) ? 1 : 0;
#endif

}

__host__ __device__
int mp_cmp(const mp_t a, const mp_t b) {
#pragma unroll
	for (int i = LIMBS - 1; i >= 0; i--) {
		if (a[i] != b[i])
			return (a[i] < b[i]) ? -1 : 1;
	}
	return 0;
}

__host__ __device__
int mp_cmp_limbs(const mp_t a, size_t limbs_a, const mp_t b, size_t limbs_b) {
#pragma unroll
	for (size_t i = max(limbs_a, limbs_b) - 1; i >= min(limbs_a, limbs_b); i--) {
		if (limbs_b < limbs_a && a[i] > 0)
			return 1;
		if (limbs_a < limbs_b && b[i] > 0)
			return -1;
	}
	for (int i = min(limbs_a, limbs_b) - 1; i >= 0; i--) {
		if (a[i] != b[i])
			return (a[i] < b[i]) ? -1 : 1;
	}
	return 0;
}

__host__ __device__
int mp_cmp_ui(const mp_t a, const mp_limb b) {
	if (a[0] > b) return 1;
	if (a[0] < b) {
#pragma unroll
		for (int i = 0; i < LIMBS; i++) {
			if (a[i] > 0) return 1;
		}
		return -1;
	}
	return 0;
}

__inline__
__host__ __device__
void mp_switch(mp_t a, mp_t b) {
	mp_limb tmp;
#pragma unroll
	for (int i = 0; i < LIMBS; i++) {
		tmp = a[i];
		a[i] = b[i];
		b[i] = tmp;
	}
}

__inline__
__host__ __device__
bool mp_isodd(mp_t a) {
	return mp_test_bit(a, 0);
}

__inline__
__host__ __device__
bool mp_iseven(mp_t a) {
	return !mp_isodd(a);
}

#ifdef __cplusplus
}
#endif

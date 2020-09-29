/*
	co-ecm
	Copyright (C) 2018  Jonas Wloka

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
			the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
			but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef MONPROD_C_MP_H
#define MONPROD_C_MP_H


#ifndef __CUDACC__ /* when compiling with g++ ... */
#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif
#endif

#include <stdio.h>
#include <stdint.h>
#include <malloc.h>
#include <cuda.h>
#include <stdbool.h>

#include "build_config.h"

#define MP_STRIDE BATCH_JOB_SIZE

#define _S_IDX(element, limb) (element + (MP_STRIDE * (limb)))

/**
 * Basic datatype for CUDA and host multi-precision numbers.
 */
typedef mp_limb mp_t[LIMBS];


typedef mp_limb mp_strided_t[MP_STRIDE * LIMBS];

/**
 * Array for basic datatypes. Needed in some places where arrays cannot be processed by Nvidias NVCC compiler.
 */
typedef mp_limb *mp_p;

#ifdef __cplusplus
extern "C" {
#endif

/***************/
/* Device code */
/***************/

#if defined(_MSC_VER)
#  define ASM asm volatile
#else
#  define ASM asm __volatile__
#endif

#if (LIMB_BITS == 32)
#define __ASM_SIZE "32"
#define __ASM_CONSTRAINT "r"

#define _PRI_ulimb PRIu32
#define _PRI_xlimb PRIx32

#elif (LIMB_BITS == 64)
#define __ASM_SIZE "64"
#define __ASM_CONSTRAINT "l"

#define _PRI_ulimb PRIu64
#define _PRI_xlimb PRIx64
#endif

#define __addc(r, a, b) ASM ("addc.u" __ASM_SIZE " %0, %1, %2;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a),__ASM_CONSTRAINT (b))
#define __add_cc(r, a, b) ASM ("add.cc.u" __ASM_SIZE " %0, %1, %2;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b))
#define __addc_cc(r, a, b) ASM ("addc.cc.u" __ASM_SIZE " %0, %1, %2;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b))
#define __subc(r, a, b) ASM ("subc.u" __ASM_SIZE " %0, %1, %2;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a),__ASM_CONSTRAINT (b))
#define __sub_cc(r, a, b) ASM ("sub.cc.u" __ASM_SIZE " %0, %1, %2;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b))
#define __subc_cc(r, a, b) ASM ("subc.cc.u" __ASM_SIZE " %0, %1, %2;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b))


#define __addcy(carry) ASM ("addc.u" __ASM_SIZE " %0, 0, 0;": "=" __ASM_CONSTRAINT (carry))
#define __addcy2(carry) ASM ("addc.cc.u" __ASM_SIZE " %0, %0, 0;": "+" __ASM_CONSTRAINT(carry))

#define __mul_lo(r, a, b) ASM("mul.lo.u" __ASM_SIZE " %0, %1, %2;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a),__ASM_CONSTRAINT (b))
#define __mul_hi(r, a, b) ASM("mul.hi.u" __ASM_SIZE " %0, %1, %2;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a),__ASM_CONSTRAINT (b))

#define __mad_lo(r, a, b, c) ASM("mad.lo.u" __ASM_SIZE " %0, %1, %2, %3;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b), __ASM_CONSTRAINT (c))
#define __mad_lo_cc(r, a, b, c) ASM("mad.lo.cc.u" __ASM_SIZE " %0, %1, %2, %3;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b), __ASM_CONSTRAINT (c))
#define __mad_hi(r, a, b, c) ASM("mad.hi.u" __ASM_SIZE " %0, %1, %2, %3;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b), __ASM_CONSTRAINT (c))
#define __mad_hi_cc(r, a, b, c) ASM("mad.hi.cc.u" __ASM_SIZE " %0, %1, %2, %3;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b), __ASM_CONSTRAINT (c))

#define __madc_hi(r, a, b, c) ASM("madc.hi.u" __ASM_SIZE " %0, %1, %2, %3;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b), __ASM_CONSTRAINT (c))

#define __madc_lo_cc(r, a, b, c) ASM("madc.lo.cc.u" __ASM_SIZE " %0, %1, %2, %3;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b), __ASM_CONSTRAINT (c))
#define __madc_hi_cc(r, a, b, c) ASM("madc.hi.cc.u" __ASM_SIZE " %0, %1, %2, %3;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b), __ASM_CONSTRAINT (c))



#if (LIMB_BITS == 32)
#define __shf_r_clamp(r, a, b, c) ASM("shf.r.clamp.b" __ASM_SIZE " %0, %1, %2, %3;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b), __ASM_CONSTRAINT (c))
#define __shf_l_clamp(r, a, b, c) ASM("shf.l.clamp.b" __ASM_SIZE " %0, %1, %2, %3;": "=" __ASM_CONSTRAINT (r): __ASM_CONSTRAINT (a), __ASM_CONSTRAINT (b), __ASM_CONSTRAINT (c))
#endif



/**
 * Print a mp_t number.
 * @param a	Number to print.
 */
__host__ __device__
void mp_print(const mp_t a);

/**
 * Print a mp_t number in hexadecimal.
 * @param a	Number to print.
 */
void mp_print_hex(const mp_t a);


void mp_print_hex_limbs(const mp_t a, size_t limbs);

/**
 * Allocate space for a mp_t on the CUDA device.
 *
 * @param a		Output parameter for the device memory address.
 */
__host__
void mp_dev_init(mp_p *a);

/**
 * Allocate space for a variable size mp_t on the CUDA device.
 *
 * @param a		Output parameter for the device memory address.
 * @param limbs	Number of limbs to allocate for this number.
 */
__host__
void mp_dev_init_limbs(mp_p *a, size_t limbs);

/**
 * Free a mp_t number.
 * @deprecated
 *
 * @param a		Number to deallocate.
 */
__host__ __device__
void mp_free(mp_t a);

/**
 * Set a mp_t to the value of a mp_limb.
 *
 * @param a		Number to set to \p s
 * @param s 	Value to set \p a to
 */
__host__ __device__
void mp_set_ui(mp_t a, mp_limb s);

/**
 * Set a := b.
 *
 * @param a
 * @param b
 */
__host__ __device__
void mp_copy(mp_t a, const mp_t b);

/**
 * Copy a mp_t number to the CUDA device.
 *
 * @param dev_a		Pointer to number in device memory.
 * @param b			Number to copy to the device.
 */
__host__
void mp_copy_to_dev(mp_p dev_a, const mp_t b);

/**
 * Copy variable size mp_t to the CUDA device.
 *
 * @param dev_a 	Pointer to number in device memory.
 * @param b 		Number to copy.
 * @param limbs 	Number of limbs in \p b.
 */
__host__
void mp_copy_to_dev_limbs(mp_p dev_a, const mp_t b, const size_t limbs);

/**
 * Copy mp_t number from device to host memory.
 *
 * @param a 		Host memory destination.
 * @param dev_b 	Device memory source.
 */
__host__
void mp_copy_from_dev(mp_t a, const mp_p dev_b);

/**
 * Set r := a * s;
 *
 * @param r
 * @param a
 * @param s
 * @return 	MSB of r exceeding the number of limbs in r.
 */
__host__ __device__
mp_limb mp_mul_ui(mp_t r, const mp_t a, const mp_limb s);

/**
 * Set r := LSB(a * b).
 * @param r
 * @param a
 * @param b
 * @return 	MSB of r exceeding the number of limbs in r.
 */
__host__ __device__
mp_limb mp_mul_limb(mp_limb *r, mp_limb a, mp_limb b);

/**
 * Set r := a + b
 *
 * @param r
 * @param a
 * @param b
 * @return 	Carry of a + b exceeding the number of limbs in r.
 */
__host__ __device__
mp_limb mp_add(mp_t r, const mp_t a, const mp_t b);

/**
 * Set r := a + b mod n
 *
 * @param r
 * @param a
 * @param b
 * @param n
 */
__host__ __device__
void mp_add_mod(mp_t r, const mp_t a, const mp_t b, const mp_t n);

/**
 * Add \p s to the \p limb -th limb of a, return carry of this limb only.
 *
 * Does not propagate carry to higher limbs.
 *
 * @param a
 * @param s
 * @param limb
 * @return
 */
__host__ __device__
mp_limb mp_limb_addc(mp_t a, const mp_limb s, const size_t limb);

/**
 * Set r:= a + b. Return carry.
 *
 * Propagates carry up to higher limbs.
 *
 * @param r
 * @param a
 * @param b
 * @return Carry exceedingt the number of limbs in r.
 */
__host__ __device__
mp_limb mp_add_ui(mp_t r, const mp_t a, const mp_limb b);

/**
 * Set r := a - b.
 *
 * @param r
 * @param a
 * @param b
 * @return Carry exceedingt the number of limbs in r.
 */
__host__ __device__
mp_limb mp_sub(mp_t r, const mp_t a, const mp_t b);

/**
 * Set r := a - b mod n.
 *
 * Effectively sets r := a + (n - b).
 * @param r
 * @param a
 * @param b
 * @param n
 */
__host__ __device__
void mp_sub_mod(mp_t r, const mp_t a, const mp_t b, const mp_t n);

/**
 * Set r := a - s.
 *
 * @param r
 * @param a
 * @param s
 */
__host__ __device__
void mp_sub_ui(mp_t r, const mp_t a, const mp_limb s);

/**
 * Set r := a * b.
 *
 * Discards any result that exceeds the number of limbs in r.
 *
 * @param r
 * @param a
 * @param b
 */
__host__ __device__
void mp_mul(mp_t r, const mp_t a, const mp_t b);

/**
 * Compare a and b.
 *
 * @param a
 * @param b
 * @return 0 if a == b, -1 if a < b, 1 if a > b
 */
__host__ __device__
int mp_cmp(const mp_t a, const mp_t b);

/**
 * Return 1 if a > b
 */
__host__ __device__
int mp_gt(const mp_t a, const mp_t b);

/**
 * Compare a and b.
 *
 * @param a
 * @param limbs_a Number of limbs in a
 * @param b
 * @param limbs_b Number of limbs in b
 * @return 0 if a == b, -1 if a < b, 1 if a > b
 */
__host__ __device__
int mp_cmp_limbs(const mp_t a, size_t limbs_a, const mp_t b, size_t limbs_b);

/**
 * Compare a and b.
 *
 * @param a
 * @param b
 * @return 0 if a == b, -1 if a < b, 1 if a > b
 */
__host__ __device__
int mp_cmp_ui(const mp_t a, const mp_limb b);

/**
 * Shift a by \p limbs to the left.
 * @param a
 * @param limbs
 */
__host__ __device__
void mp_sl_limbs(mp_t a, size_t limbs);

/**
 * Shift a by \p limbs to the right.
 * @param a
 * @param limbs
 */
__host__ __device__
void mp_sr_limbs(mp_t a, size_t limbs);

/**
 * Switch the values of a and b.
 * @param a
 * @param b
 */
__host__ __device__
void mp_switch(mp_t a, mp_t b);

__host__ __device__
bool mp_iseven(mp_t a);


#ifdef __cplusplus
}
#endif


/**
 * Set bit number \p bit in a to 1.
 */
#define mp_set_bit(a, bit) ((a)[(bit)/LIMB_BITS] |= ((mp_limb_t)1 << ((bit)%LIMB_BITS)))

/**
 * Return whether bit number \p bit in a is set.
 */
//#define mp_test_bit(a, bit) ((((a)[(bit)/LIMB_BITS]) >> ((bit)%LIMB_BITS)) & 1)
#define mp_test_bit(a, bit) !!(((a)[(bit)/LIMB_BITS]) & (1 << ((bit)%LIMB_BITS)))


#endif //MONPROD_C_MP_H

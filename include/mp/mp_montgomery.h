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

#ifndef MONPROD_C_MP_MONTGOMERY_H
#define MONPROD_C_MP_MONTGOMERY_H

#include "mp.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * Montgomery parameters info struct.
 *
 * Used wherever montgomery arithmetic is used to cache values.
 */
typedef struct {
	mp_t R2;	/**< R squared */
	mp_t n;		/**< Modulus */
	mp_limb mu;	/**< R2 mod r */
} mon_info;

typedef struct {
	mp_strided_t R2;		/**< R squared */
	mp_strided_t n;			/**< Modulus */
	mp_limb mu[BATCH_JOB_SIZE];	/**< -n^{-1} mod r */
} mon_info_strided;

/**
 * Compute montgomery exponentiation.
 *
 * Sets r := b^e mod n.
 *
 * @param r
 * @param b
 * @param e
 * @param info	Montgomery parameter info struct.
 */
__host__ __device__
void mon_exp(mp_t r, const mp_t b, const mp_t e, const mon_info *info);

/**
 * Compute montgomery multiplication.
 *
 * Sets res := a * b mod n.
 *
 * @param res
 * @param a
 * @param b
 * @param info
 * @return
 */
__host__ __device__
int mon_prod(mp_t res, const mp_t a, const mp_t b, const mon_info *info);

__host__ __device__
/**
 * Compute the montgomery inverse.
 *
 * Sets res := a^(-1) mod info->n.
 * Implementation uses binary extended euclidean algorithm.
 *
 * @param res
 * @param a
 * @param info
 */
int mon_inv(mp_t res, const mp_t a, const mon_info *info); 

__host__ __device__
int mon_prod_distinct(mp_t res, const mp_t a, const mp_t b, const mon_info *info);

/**
 * Transforms \p a to Montgomery representation in \p r.
 *
 * @param r
 * @param a
 * @param info
 * @return
 */
__device__ __host__
int to_mon(mp_t r, const mp_t a, const mon_info *info);

/**
 * Transforms \p a from Montgomery representation in \r.
 *
 * @param r
 * @param a
 * @param info
 * @return
 */
__device__ __host__
int from_mon(mp_t r, const mp_t a, const mon_info *info);


/**
 * Copy Montgomery info struct to device.
 *
 * @param host_info
 * @return	Pointer to device memory containing the struct.
 */
mon_info *mon_info_copy_to_dev(mon_info *host_info);

/**
 * Compute Montgomery parameters for given n.
 *
 * @param n		Modulus
 * @param info	Populated Montgomery parameter info struct.
 * @return
 */
int mon_info_populate(mp_t n, mon_info *info);

/**
 * Print Montgomery parameter info struct.
 *
 * @param info	Pointer to info struct in host memory.
 */
__host__
void print_info(const mon_info *info);

/**
 * Print Montgomery parameter info struct from the CUDA device.
 *
 * @param info	Pointer to info struct in device memory.
 */
__device__
void print_info_dev(const mon_info *info);

#ifdef __cplusplus
}
#endif

#endif //MONPROD_C_MP_MONTGOMERY_H

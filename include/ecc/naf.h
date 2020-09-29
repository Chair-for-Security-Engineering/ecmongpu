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

#ifndef CO_ECM_NAF_H
#define CO_ECM_NAF_H

#include "mp/mp.h"
#include "build_config.h"
#include <gmp.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Macro to help with computing the array storage of precomputed values to be used with NAF scalar multiplication.
 *
 * Only odd values are precomputed, the lowest number is 1, mapped to array index 0. 3 is mapped to array index 1,
 */
//#define __naf_to_index(i) (((i+1)/2)-1)
#define __naf_to_index(i) ((i)>>1)

/**
 * Number of precomputed points for w-NAF.
 *
 * All odd point lP with l < 2^(w-1), thus 2^(w-2) = (2^w)/4
 */
#define NAF_N_PRECOMPUTED ((1 << NAF_WINDOW_SIZE) / 4)

/**
 * Maximum l for precomputed points lP for w-NAF.
 *
 * All odd Points lP with l < 2^(w-1) are precomputed.
 */
#define NAF_MAX_PRECOMPUTED (((1 << NAF_WINDOW_SIZE) / 2) - 1)

/**
 * Type for a NAF number.
 *
 * Elements of the array are single digits of the NAF scalar.
 */
typedef uint8_t naf_limb;
typedef naf_limb *naf_t;

/**
 * Coverts a mpz_t type to NAF format.
 *
 * Processes (a few bits less than a) mp_limb at a time to reduce the amount of bitshifts on the large input number.
 *
 * @param NAF Output parameter.
 * @param digits  The maximum number of output digits allowed by \p naf (e.g. the size of the array).
 * @param s Scalar to transform to NAF
 * @param w NAF Window size.
 * @return Number of digits in the NAF scalar or `-1` if an error occured or the scalar would be larger than \p digits.
 */
int to_naf(naf_t naf, size_t digits, mpz_t s, int w);

/**
 * Converts a NAF number to an mpz_t.
 *
 * @param s GMP's mpz_t output parameter.
 * @param naf The scalar in NAF form to convert.
 * @param digits Number of digits in the NAF scalar.
 */
void from_naf(mpz_t s, naf_t naf, size_t digits);

/**
 * Prints a scalar in NAF form.
 *
 * @param a Scalar in NAF form.
 * @param digits Number of digits in NAF scalar.
 */
__host__ __device__
void print_naf(naf_t a, size_t digits);


#ifdef __cplusplus
}
#endif

#endif //CO_ECM_NAF_H

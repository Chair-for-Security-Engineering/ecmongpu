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

#ifndef MONPROD_C_GMP_CONVERSION_H
#define MONPROD_C_GMP_CONVERSION_H

#include "mp/mp.h"
#include "gmp.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Transform a number from GMPs mpz_t to this library's mp_t.
 *
 * @param a Output parameter MP number.
 * @param b	Number to transform.
 */
void mpz_to_mp(mp_t a, const mpz_t b);

/**
 * Transform a number from GMPs mpz_t to this library's mp_t.
 *
 * @param a 	Output parameter MP number.
 * @param b		Number to transform.
 * @param limbs	Maximum size of the output number a
 */
void mpz_to_mp_limbs(mp_t a, const mpz_t b, const size_t limbs);

/**
 * Transform a number from this library's mp_t to GMPs mpz_t.
 *
 * @param a	Number to transform.
 * @param b Output parameter MP number.
 */
void mp_to_mpz(mpz_t a, const mp_t b);

/**
 * Print with printf syntax a number from this library's mp_t form.
 *
 * Actually transforms to GMPs representation and uses gmp_printf.
 *
 * @param format	printf-style format string. Use "%Zi" for mp_t.
 * @param a			mp_t number to print
 */
void mp_printf(const char *format, const mp_t a);

#ifdef __cplusplus
}
#endif

#endif //MONPROD_C_GMP_CONVERSION_H

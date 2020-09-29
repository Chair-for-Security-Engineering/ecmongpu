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
#ifndef ECC_TWISTED_EDWARDS_H
#define ECC_TWISTED_EDWARDS_H

#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include <gmp.h>
#include "ecc/naf.h"

#ifdef __cplusplus
extern "C" {
#endif



/**
 * Struct containing a curve in Twisted Edwards form.
 */
typedef struct __curve_tw_ed {
	mp_t d; /**< Curve parameter d */
	mp_t k; /**< Precomputed k = 2*d */
	mon_info *info;
} curve_tw_ed;

typedef struct __curve_tw_ed_strided {
  mp_strided_t d;
  mp_strided_t k;
} curve_tw_ed_strided;


typedef struct {
  mon_info mon_info;
  curve_tw_ed curve;
} shared_mem_cache;


typedef struct {
	mpz_t x;
	mpz_t y;
	pthread_mutex_t mutex;
} point_gkl2016;

/**
 * Struct containing extended projective point coordinates on a Twisted Edwards curve.
 */
typedef struct __point_tw_ed {
	mp_t x; /**< Point x coordinate */
	mp_t y;	/**< Point y coordinate */
	mp_t z;	/**< Point z coordinate */
#ifdef COORDINATES_EXTENDED
	mp_t t;	/**< Point t coordinate */
#endif
} point_tw_ed;

typedef struct __point_tw_ed_strided {
	mp_strided_t x; /**< Point x coordinate */
	mp_strided_t y;	/**< Point y coordinate */
	mp_strided_t z;	/**< Point z coordinate */
#ifdef COORDINATES_EXTENDED
	mp_strided_t t;	/**< Point t coordinate */
#endif
} point_tw_ed_strided;


__host__ __device__
void tw_ed_point_set_id(point_tw_ed *p);

__host__ __device__
void tw_ed_point_invert(point_tw_ed *p, const mon_info *info);

__host__ __device__
void tw_ed_to_reg(point_tw_ed *p, const mon_info *info);

__host__ __device__
void tw_ed_point_invert_precomp(point_tw_ed *p, const mon_info *info);

/**
 * Add two points on a Twisted Edwards curve.
 *
 * @param r		Return parameter for the result of the addition.
 * @param op1 	First point to add.
 * @param op2 	Second point to add.
 * @param curve Curve both points belong to.
 * @param info 	Montgomery parameters struct associated with the curve.
 * @return 		Always 0.
 */
__host__ __device__
int tw_ed_add(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
			  const mon_info *info, bool extend);

__host__ __device__
int tw_ed_sub(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
			  const mon_info *info, bool extend);

__host__ __device__
int tw_ed_add_precomp(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
			  const mon_info *info, bool extend);

/**
 * Add two points on a "a=-1" Twisted Edwards curve.
 *
 * See https://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-4 for more information
 * on the algorithm. **Caution:** The formula is not unified and will give wrong results if op1 == op2.
 *
 * @param r		Return parameter for the result of the addition.
 * @param op1 	First point to add.
 * @param op2 	Second point to add.
 * @param curve Curve both points belong to.
 * @param info 	Montgomery parameters struct associated with the curve.
 * @return 		Always 0.
 */
int tw_ed_add_hwcd4(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
					const mon_info *info);

/**
 * Double a point on a "a=-1" Twisted Edwards curve.
 *
 * See https://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#doubling-dbl-2008-hwcd for more information
 * on the algorithm.
 *
 * @param r		Return parameter for the doubled point.
 * @param op	Point to double.
 * @param curve Curve the point belongs to.
 * @param info	Montgomery parameter struc associated with the curve.
 * @return 		Always 0.
 */
__host__ __device__
int tw_ed_double(point_tw_ed *r, const point_tw_ed *op, const curve_tw_ed *curve, const mon_info *info, bool extend);


__host__ __device__
int tw_ed_triple(point_tw_ed *r, const point_tw_ed *op, const curve_tw_ed *curve, const mon_info *info, bool extend);

__host__ __device__
int tw_ed_double_cs(point_tw_ed *r, const point_tw_ed_strided *op, const size_t op_elem, const curve_tw_ed *curve, const mon_info *info);

/**
 * Fill a curve struct with parameters \p a, \p d and set the Montgomery parameters of the curve.
 *
 * Memory should be allocated by the caller.
 *
 * @param curve	Return parameter with the filled curve struct.
 * @param a		Curve parameter a.
 * @param d		Curve parameter d.
 * @param info	Montgomery parameters.
 * @return		Always 0.
 */
int tw_ed_init_curve(curve_tw_ed *curve, mp_t a, mp_t d, mon_info *info);

/**
 * Copy a curve from the Host do a CUDA device.
 *
 * @param curve Curve to copy
 * @return 		Pointer to the curve in Device memory.
 */
__host__
curve_tw_ed *tw_ed_copy_curve_to_dev(const curve_tw_ed *curve);

/**
 * Copy a point to a CUDA device.
 *
 * @param dev_point 	Memory address in device memory where the point should be copied to.
 * @param host_point 	Pointer to host memory containing the point.
 * @param dev_curve 	Pointer to device memory containing the curve struct.
 * @return 				Always 0
 */
__host__
int tw_ed_copy_point_to_dev(const point_tw_ed *dev_point, const point_tw_ed host_point, curve_tw_ed *dev_curve);

/**
 * Copy a point from a CUDA device to host memory.
 *
 * @param host_point 	Pointer to host memory where the point is copied.
 * @param dev_point 	Memory address in device memory containing the point.
 * @return 				Always 0
 */
__host__
int tw_ed_copy_point_from_dev(point_tw_ed *host_point, point_tw_ed *dev_point);

/**
 * Scale a Point (x, y, z, t) from projective coordinates, s.t. z == 1, e.g. (x/z, y/z, 1, x*y)
 * @param r		Return parameter for scaled point.
 * @param p		Point to scale.
 * @param info	Montgomery parameters according to Point.
 */
__host__ __device__
bool tw_ed_scale_point(point_tw_ed *r, const point_tw_ed *p, const mon_info *info);

/**
 * Clean up memory used for the curve.
 *
 * @param curve
 * @return
 */
int tw_ed_free_curve(curve_tw_ed *curve);

/**
 * Print out a curve and its parameters.
 *
 * @param c		The Curve.
 */
void tw_ed_print_curve(const curve_tw_ed *c);

/**
 * Prints a point in Montgomery format and "regular" format.
 *
 * @param p		Point to print.
 * @param info	Montgomery parameters according to point.
 */
void tw_ed_print_point(const point_tw_ed *p, const mon_info *info);

/**
 * Copy point, e.g. assign \p dest a new value.
 *
 * @param dest	Destination
 * @param src	Source
 * @return		Always 0.
 */
__host__ __device__
int tw_ed_copy_point(point_tw_ed *dest, const point_tw_ed *src);

__host__ __device__
int tw_ed_copy_point_sc(point_tw_ed_strided *dest, const size_t d_elem, const point_tw_ed *src);

__host__ __device__
int tw_ed_copy_point_cs(point_tw_ed *dest, const point_tw_ed_strided *src, const size_t s_elem);
__host__ __device__
int tw_ed_copy_point_ss(point_tw_ed_strided *dest, const size_t d_elem, const point_tw_ed_strided *src, const size_t s_elem);

/**
 * Copy curve from \p src to \p dest.
 *
 * @param dest	Point to where the curve should be written.
 * @param src	Point to be copied to \p dest.
 * @return		Always 0.
 */
__host__ __device__
int tw_ed_copy_curve(curve_tw_ed *dest, const curve_tw_ed *src);

__host__ __device__
int tw_ed_copy_curve_sc(curve_tw_ed_strided *dest, const curve_tw_ed *src);

__host__ __device__
int tw_ed_copy_curve_cs(curve_tw_ed *dest, const curve_tw_ed_strided *src);


/**
 * Scalar multiplication of point \p op times \p scalar.
 *
 * Computes r := [scalar]op.
 * @param r					Return parameter.
 * @param op				Point to multiply.
 * @param curve				Curve the point lies on.
 * @param scalar			Scalar to multiply by, type mp_p
 * @param scalar_bitlength	Bitlength of the scalar.
 * @param info				Montgomery parameter info struct.
 * @return					Always 0.
 */
__host__ __device__
int tw_ed_smul(point_tw_ed *r, const point_tw_ed *op, const curve_tw_ed *curve, const mp_p scalar,
			   const unsigned int scalar_bitlength, const mon_info *info);


/* Helper for ECM */

/**
 * Check whether a point is on a curve, i.e. fulfills the curve equation.
 *
 * @param p		Point to check.
 * @param curve	Curve to check whether the point is on.
 * @param info	Montgomery parameter info struct.
 * @return 		0 if not on curve, else other.
 */
__host__ __device__
int tw_ed_point_on_curve(const point_tw_ed *p, const curve_tw_ed *curve, const mon_info *info);

/**
 * Construct a random "a = -1" Twisted Edwards curve in \p curve with point \p p1 on the curve.
 *
 * @param p1 		Output parameter for the Point.
 * @param curve 	Output parameter for the Curve.
 * @param info 		Montgomery parameter info struct.
 * @param gmprand 	GMP random number object.
 */
void tw_ed_random_curve_naive(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand, void* data);

/**
 * \copydoc tw_ed_random_curve_naive
 *
 * Uses Curve construction with parameter j = 1 from
 * A. Gélin, T. Kleinjung, and A. K. Lenstra, “Parametrizations for Families of ECM-friendly curves,” 1092, 2016.
 */
void tw_ed_random_curve_gkl2016_j1(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand, void* data);

/**
 * \copydoc tw_ed_random_curve_naive
 *
 * Uses Curve construction with parameter j = 4 from
 * A. Gélin, T. Kleinjung, and A. K. Lenstra, “Parametrizations for Families of ECM-friendly curves,” 1092, 2016.
 */
void tw_ed_random_curve_gkl2016_j4(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand, void* data);

/**
 * Type for Curve generation function pointer.
 */
typedef void (*job_generator)(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand, void* data);

/**
 * Array of Curve generators. Used to select the curve generation to use from the CLI.
 *
 * Add new Curve construction functions here, their name below and increase the count, and they will be available from
 * the CLI.
 */
extern job_generator const job_generators[];

/**
 * Names for Curve generation algorithms in job_generators. Shown in help message and log.
 */
extern const char *const job_generators_names[];

/**
 * Length of the job_generators function pointer array.
 */
extern const int job_generators_len;



#ifdef __cplusplus
}
#endif

#endif /* ECC_TWISTED_EDWARDS_H */

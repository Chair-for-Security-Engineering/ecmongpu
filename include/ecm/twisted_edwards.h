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
#ifndef ECM_TW_ED_EXTENDED_H
#define ECM_TW_ED_EXTENDED_H

#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include <gmp.h>
#include "ecc/naf.h"

#ifdef __cplusplus
extern "C" {
#endif

extern pthread_mutex_t mutex_gmp_rand;

/**
 * Compute the scalar multiplication (using NAF-form) for a _single_ job on the point p.
 * Assumes that precomputed_strided is filled with small multiples of point p.
 *
 * @param p 					Point to multiply (input and output)
 * @param curve 				Curve the point is on
 * @param precomputed_strided 	Containing precomputed small multiples of p (in strided form)
 * @param info 					Montgomery info
 * @param scalar 				Scalar to multiply by (in NAF-form)
 * @param scalar_size			Number of digits in the (NAF-)scalar
 * @param myjob					Job to work on (needed for indexing the precomputed strided array)
 *
 * @return						Number of digits processed from the NAF-scalar
 */
__host__ __device__
size_t tw_ed_smul_naf(point_tw_ed *p, 
    const curve_tw_ed *curve, 
    const point_tw_ed_strided precomputed_strided[NAF_N_PRECOMPUTED],
    const mon_info *info, 
    const naf_t scalar, 
    const size_t scalar_size, 
    const size_t myjob);


/**
 * Optimizes a precomputed point.
 * Actual implementation depends on the selected coordinate system.
 *
 * @param r			Return parameter of the optimized point
 * @param curve 	Curve the point is on
 * @param p			Point to optimize
 * @param info		Montgomery info
 *
 */
__host__ __device__
void tw_ed_optimize_precomp(point_tw_ed *r, const point_tw_ed *p, curve_tw_ed *curve, mon_info *info);


/**
 * Callback to be called from CUDA runtime once all GPU computation has finished for stage1.
 *
 * @param batch			Pointer to the batch_job_naf for this stream.
 */
__host__
void batch_finished_cb_stage1(batch_job_naf *batch);


/**
 * Callback to be called from CUDA runtime once all GPU computation has finished for stage2.
 *
 * @param batch			Pointer to the batch_job_naf for this stream.
 */
__host__
void batch_finished_cb_stage2(batch_job_naf *batch);

#ifdef __cplusplus
}
#endif


#endif

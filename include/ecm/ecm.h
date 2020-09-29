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

#ifndef CUDAECM_H
#define CUDAECM_H
#include <gmp.h>
#include <pthread.h>

#include "ecm/batch.h"
#include "ecm/factor_task.h"
#include "ecm/twisted_edwards.h"

#ifdef __cplusplus
 extern "C" {
#endif

typedef struct _stage2_results_s {
  batch_job_naf *batch;
  mp_t *results;
} stage2_results_s;


extern pthread_mutex_t mutex_outfile;

/**
 * CUDA kernel that computes scalar NAF multiplication of all points in a batch.
 *
 * @param batch			Batch job to work on.
 * @param scalar		Scalar in NAF form.
 * @param naf_digits	Number of digits in the NAF scalar.
 */
__global__
void cuda_tw_ed_smul_naf_batch(batch_job_data_naf *batch, naf_t scalar, size_t naf_digits);

/**
 * CUDA kernel to compute scalar multiplication for all jobs in a batch.
 *
 * @param batch				Batch to work on.
 * @param scalar			Scalar to multiply .point member with.
 * @param scalar_bitlength	Bitlength of the scalar.
 */
__global__
void cuda_tw_ed_smul_batch(batch_job *batch, const mp_p scalar, const unsigned int scalar_bitlength);


/**
 * CUDA kernel to compute the second stage of ECM for all jobs in a batch.
 *
 * @param batch				    Batch to work on.
 * @param globals			    Structure with global stage 2 values (Prime Bitfield, Babysteps/Giantsteps values, ...)
 * @param babysteps_y 		    Strided array containing the y-coordinates of babysteps
 * @param babysteps_y_tmp 	    Strided array for storing temporary scaled y-coordinates of babysteps
 * @param babysteps_z		    Strided array containing z-coordinates of babysteps
 * @param babysteps_t		    Strided array containing t-coordinates of babysteps
 * @param giantsteps_y		    Strided array to store y-coordinates of giantsteps
 * @param giantsteps_z		    Strided array to store z-coordinates of giantsteps
 * @param giantsteps_t		    Strided array to store t-coordinates of giantsteps
 * @param bufsize_giantsteps    Size of the giantsteps arrays
 */
__global__
void cuda_tw_ed_stage2(batch_job_data_naf *batch, stage2_global *globals,
					   mp_strided_t *babysteps_y, mp_strided_t *babysteps_y_tmp, mp_strided_t *babysteps_z, mp_strided_t *babysteps_t,
					   mp_strided_t *giantsteps_y, mp_strided_t *giantsteps_z, mp_strided_t *giantsteps_t,
					   size_t bufsize_giantsteps);

/**
 * CUDA kernel to check for each job in batch whether the point is on the curve.
 *
 * Populates the struct member .on_curve.
 * @param batch		Batch to work on.
 */
__global__
void cuda_tw_ed_point_on_curve(batch_job_data *batch);

/**
 * CUDA kernel to check for each job in batch whether the point is on the curve.
 * NAF version.
 *
 * Sets the job struct member .on_curve to 0 if not on curve or 1 if on curve.
 *
 * @param batch		Batch to work on.
 */
__global__
void cuda_tw_ed_point_on_curve_naf(batch_job_data_naf *batch);

/**
 * CUDA function precomputing small multiples for NAF-/chain point-multipliplication for a _single_ job
 *
 * @param cache 	Pointer to the shared_memory used for user-defined caching of global memory
 * @param batch		Batch to work on
 * @param myjob		Which job to work on in the batch
 */
__device__
void tw_ed_naf_precompute(shared_mem_cache *cache, batch_job_data_naf *batch, int myjob);



#ifdef __cplusplus
};
#endif

#endif

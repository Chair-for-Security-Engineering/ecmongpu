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

#ifndef CO_ECM_BATCH_H
#define CO_ECM_BATCH_H

#include "ecc/twisted_edwards.h"
#include "ecc/naf.h"
#include "ecm/factor_task.h"
#include "config/config.h"
#include "build_config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Struct holding a single batch jobs info
 */
typedef struct _batch_job_data {
	mon_info mon_info;
	curve_tw_ed curve;
	point_tw_ed point;
	point_tw_ed basepoint;
	int8_t on_curve;
} batch_job_data;

/**
 * Struct holding a collection of batch jobs and pointers to associated data on host.
 */
typedef struct {
	batch_job_data job[BATCH_JOB_SIZE];
	int tasks_id[BATCH_JOB_SIZE];
	run_config config;
} batch_job;


/**
 * Struct holding a single batch jobs info in naf version, including precomputed points.
 */

typedef struct _batch_job_data_naf {
	mon_info_strided mon_info_strided;

	curve_tw_ed_strided curve_strided;

	point_tw_ed_strided point_strided;

	point_tw_ed_strided precomputed_strided[NAF_N_PRECOMPUTED];

	int8_t on_curve[BATCH_JOB_SIZE];
	mp_t stage2_result[BATCH_JOB_SIZE];
} batch_job_data_naf;

/**
 * Struct holding a batch job for use with NAF multiplication.
 */
typedef struct {
	size_t n_jobs;
	int cuda_blocks;
	batch_job_data_naf job;
	int tasks_id[BATCH_JOB_SIZE];
	struct {
		mp_strided_t *y;
		mp_strided_t *y_tmp;
		mp_strided_t *z;
		mp_strided_t *t;
	} babysteps;
	struct {
		mp_strided_t *y;
		mp_strided_t *z;
		mp_strided_t *t;
		size_t bufsize;
	} giantsteps;

	run_config config;

	int device;

} batch_job_naf;

/**
 * Data pointers for a batch in naf form, containing pointers to host and device memory
 */
typedef struct _batch_ptr_naf {
	batch_job_naf **host;
	batch_job_naf **dev;
} batch_naf;


/**
 * Print out all jobs in a batch with curve and points.
 *
 * @param batch		Batch to work on.
 */
__host__
void print_batch(batch_job *batch);

/**
 * Print out all jobs in a batch with curve and points.
 * NAF version.
 *
 * @param batch		Batch to work on.
 */
__host__
void print_batch_naf(batch_job_naf *batch);


/**
 * Prepares all jobs in a batch.
 *
 * Computes random curves and starting points for each job, using the \p gmprand RNG.
 *
 * @param batch 		Batch to work on.
 * @param gmprand 		GMP's RNG to gather randomness from during curve generation.
 * @param stream 		The CUDA stream this batch is associated with.
 * @param generate_job 	Function to generate the curve and point.
 */
__host__
void compute_batch_job(factor_task task, run_config, batch_job_naf *batch, size_t job);


/**
 * Allocate memory on host and device for the given batch.
 * @param config	configuration to attach to this batch.
 * @param batch		batch to save pointers to allocated memory in.
 */
__host__
void batch_allocate(run_config config, batch_naf *batch);

#ifdef __cplusplus
}
#endif

#endif //CO_ECM_BATCH_H

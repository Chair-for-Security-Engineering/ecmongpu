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

#ifndef ECM_STAGE1_H
#define ECM_STAGE1_H

#include "config/config.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run stage 1 of ECM with the given configuration and batch.
 *
 * Computes random curves and starting points P modulo the numbers n popped from the stage1 heap in config.
 *
 * @param batch		The batch with allocated memory to use.
 * @param config	The configuration object.
 */
void ecm_stage1(run_config config, batch_naf *batch, size_t stream);

/**
 * Run the initializiation for stage1. Has to be called only once per program run.
 *
 * @param config 	Runtime configuration
 */
void ecm_stage1_init(run_config config);

/**
 * CUDA kernel to compute the stage1 scalar multiplication (using NAF-multiplication) for each job in a batch.
 *
 * @param batch			The batch to work on
 * @param scalar		Scalar in NAF-form
 * @param naf_digits	Number of digits in the NAF-scalar
 *
 */
__global__
void cuda_tw_ed_smul_naf_batch(batch_job_data_naf batch[BATCH_JOB_SIZE],
							   naf_t scalar, size_t naf_digits);

#ifdef __cplusplus
}
#endif

#endif /* ECM_STAGE1_H */

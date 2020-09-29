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

#ifndef ECM_STAGE2_H
#define ECM_STAGE2_H

#include "ecm/factor_task.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Run stage 2 of ECM on the given batch.
 *
 * Takes the point P of from each job, and computes [b]P for all b1 <= b <= b2.
 * In each step, the point P is updated and only the difference of two primes d = p_2 - p_1 is added to the point.
 *
 * @param batch		The batch with allocated memory already processed by stage1.
 * @param config	The configuration object.
 */
void ecm_stage2(run_config config, batch_naf *batch, size_t stream);

/**
 * Run stage 2 initialization. Has to be called only once during program run.
 *
 * @param config	Runtime configuration object
 */
void ecm_stage2_init(run_config config);

/**
 * Run the initializiation (GPU memory allocation) for stage 2 for batch.
 * Has to be called once per batch that the stage2 kernel is run on.
 *
 * @param config	Runtime configuration
 * @param batch		Batch to allocate memory to
 */
void ecm_stage2_initbatch(run_config config, batch_naf *batch);

#ifdef __cplusplus
}
#endif

#endif /* ECM_STAGE2_H */

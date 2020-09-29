#include <stdlib.h>
#include <getopt.h>
#include <gmp.h>
#include <ecm/batch.h>
#include <cuda_runtime.h>
#include <cudautil.h>

#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "ecc/twisted_edwards.h"
#include "ecm/batch.h"
#include "ecm/factor_task.h"
#include "ecm/ecm.h"
#include "ecc/naf.h"

#include "log.h"
#include "ecm/batch.h"
#include "ecm/stage2.h"


void stage2_copy_babysteps_to_dev(stage2_global *dst, stage2_global *src) {
	/* babysteps */
	dst->babysteps.n = src->babysteps.n;
	dst->babysteps.naf_size = (size_t *) malloc(src->babysteps.n * sizeof(size_t));
	naf_t *baby_devptrs = (naf_t *) malloc(src->babysteps.n * sizeof(naf_t));

	CUDA_SAFE_CALL(cudaMalloc((void **) &dst->babysteps.naf_size, src->babysteps.n * sizeof(size_t)));
	CUDA_SAFE_CALL(cudaMemcpy(dst->babysteps.naf_size,
							  src->babysteps.naf_size,
							  src->babysteps.n * sizeof(size_t),
							  cudaMemcpyHostToDevice));
	for (size_t i = 0; i < src->babysteps.n; i++) {
		CUDA_SAFE_CALL(cudaMalloc((void **) &(baby_devptrs[i]), src->babysteps.naf_size[i] * sizeof(naf_limb)));
		CUDA_SAFE_CALL(cudaMemcpy(baby_devptrs[i],
								  src->babysteps.naf[i],
								  src->babysteps.naf_size[i] * sizeof(naf_limb),
								  cudaMemcpyHostToDevice));
	}

	CUDA_SAFE_CALL(cudaMalloc((void **) &dst->babysteps.naf, src->babysteps.n * sizeof(naf_t)));
	CUDA_SAFE_CALL(cudaMemcpy(dst->babysteps.naf,
							  baby_devptrs,
							  src->babysteps.n * sizeof(naf_t),
							  cudaMemcpyHostToDevice));

}

void ecm_stage2_init(run_config config) {

	config->stage2.global_host = (stage2_global *) malloc(sizeof(_stage2_global));

	int w = config->stage2.window_size;

	/* Compute naf(w) */
	mpz_t gmp_w;
	mpz_init_set_ui(gmp_w, w);
	size_t naf_size = mpz_sizeinbase(gmp_w, 2) + 2;
	config->stage2.global_host->w.naf = (naf_t) malloc(naf_size * sizeof(naf_limb));
	config->stage2.global_host->w.naf_size = to_naf(config->stage2.global_host->w.naf,
													naf_size,
													gmp_w,
													NAF_WINDOW_SIZE);

	/* Determine number of babysteps */
	config->stage2.global_host->babysteps.n = 0;
	mpz_t gmp_u;
	mpz_init(gmp_u);
	for (mpz_set_ui(gmp_u, 1); mpz_cmp_ui(gmp_u, (w / 2)) <= 0; mpz_add_ui(gmp_u, gmp_u, 1)) {
		if (mpz_gcd_ui(NULL, gmp_u, w) == 1) config->stage2.global_host->babysteps.n++;
	}
	LOG_DEBUG("Using %d babysteps", config->stage2.global_host->babysteps.n);


	config->stage2.global_host->babysteps.naf_size = (size_t *) malloc(
			config->stage2.global_host->babysteps.n * sizeof(size_t));
	config->stage2.global_host->babysteps.naf = (naf_t *) malloc(
			config->stage2.global_host->babysteps.n * sizeof(naf_t));

	/* Compute babystep naf_t values */
	int i = 0;
	for (mpz_set_ui(gmp_u, 1); mpz_cmp_ui(gmp_u, (w / 2)) <= 0; mpz_add_ui(gmp_u, gmp_u, 1)) {
		if (mpz_gcd_ui(NULL, gmp_u, w) == 1) {
			LOG_VERBOSE("babystep u[%i]:\t%Zi", i, gmp_u);
			naf_size = mpz_sizeinbase(gmp_u, 2) + 2;
			config->stage2.global_host->babysteps.naf[i] = (naf_t) malloc(naf_size * sizeof(naf_limb));
			config->stage2.global_host->babysteps.naf_size[i] = to_naf(config->stage2.global_host->babysteps.naf[i],
																	   naf_size,
																	   gmp_u,
																	   NAF_WINDOW_SIZE);
			// TODO: Check return
			i++;
		}
	}


	/* Determine number of giantsteps */
	config->stage2.global_host->giantsteps_n = 0;
	mpz_t gmp_v;
	mpz_init(gmp_v);
	for (mpz_set_ui(gmp_v, config->b1 / w); mpz_cmp_ui(gmp_v, config->b2 / w + 1) <= 0; mpz_add_ui(gmp_v, gmp_v, 1)) {
		config->stage2.global_host->giantsteps_n++;
	}
	LOG_DEBUG("Using %d giantsteps", config->stage2.global_host->giantsteps_n);


	/* Make array with (giantsteps * babysteps) bits in mp_limbs */
	int combinations = config->stage2.global_host->giantsteps_n * config->stage2.global_host->babysteps.n;
	config->stage2.global_host->is_prime = (mp_limb *) calloc(((combinations + LIMB_BITS - 1) / LIMB_BITS),
															  sizeof(mp_limb));

	/* Loop over all v*w+u values and mark those that are prime */
	mpz_t tmp;
	mpz_init(tmp);
	int count = 0;
	for (mpz_set_ui(gmp_v, config->b1 / w); mpz_cmp_ui(gmp_v, config->b2 / w + 1) <= 0; mpz_add_ui(gmp_v, gmp_v, 1)) {
		for (mpz_set_ui(gmp_u, 1); mpz_cmp_ui(gmp_u, (w / 2)) <= 0; mpz_add_ui(gmp_u, gmp_u, 1)) {
			if (mpz_gcd_ui(NULL, gmp_u, w) == 1) {
				mpz_set_ui(tmp, w);
				mpz_mul(tmp, tmp, gmp_v);
				mpz_add(tmp, tmp, gmp_u);
				bool prime = (mpz_probab_prime_p(tmp, 10) > 0);
				mpz_sub(tmp, tmp, gmp_u);
				mpz_sub(tmp, tmp, gmp_u);
				prime |= (mpz_probab_prime_p(tmp, 15) > 0);
				if (prime) {
					mp_set_bit(config->stage2.global_host->is_prime, count);
				}
				count++;
			}
		}
	}


	/* Alloc and copy of global for device */
	stage2_global global_dev;
	global_dev.w.naf_size = config->stage2.global_host->w.naf_size;
	global_dev.giantsteps_n = config->stage2.global_host->giantsteps_n;

	for (int dev = 0; dev < config->devices; dev++) {
		cudaSetDevice(dev);
		LOG_INFO("[Device %i] Stage 2 Initialization...", dev);

		/* Copy bitfield */
		CUDA_SAFE_CALL(cudaMalloc((void **) &global_dev.is_prime,
								  ((combinations + LIMB_BITS - 1) / LIMB_BITS) * sizeof(mp_limb)));
		CUDA_SAFE_CALL(cudaMemcpy(global_dev.is_prime,
								  config->stage2.global_host->is_prime,
								  ((combinations + LIMB_BITS - 1) / LIMB_BITS) * sizeof(mp_limb),
								  cudaMemcpyHostToDevice));

		/* Copy w NAF */
		CUDA_SAFE_CALL(cudaMalloc((void **) &global_dev.w.naf,
								  config->stage2.global_host->w.naf_size * sizeof(naf_limb)));
		CUDA_SAFE_CALL(cudaMemcpy(global_dev.w.naf,
								  config->stage2.global_host->w.naf,
								  config->stage2.global_host->w.naf_size * sizeof(naf_limb),
								  cudaMemcpyHostToDevice));


		stage2_copy_babysteps_to_dev(&global_dev, config->stage2.global_host);
		CUDA_SAFE_CALL(cudaMalloc((void **) &config->dev_ctx[dev].stage2.global_dev, sizeof(stage2_global)));
		CUDA_SAFE_CALL(cudaMemcpy(config->dev_ctx[dev].stage2.global_dev,
								  &global_dev,
								  sizeof(stage2_global),
								  cudaMemcpyHostToDevice));
	}

	LOG_DEBUG("Num babysteps %i", config->stage2.global_host->babysteps.n);
}

void ecm_stage2_initbatch(run_config config, batch_naf *batch) {
	for (int dev = 0; dev < config->devices; dev++) {
		cudaSetDevice(dev);

		int offset = dev * config->n_cuda_streams;
		/* Allocate babysteps memory */
		for (int stream = 0; stream < config->n_cuda_streams; stream++) {
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &batch->host[offset + stream]->babysteps.y,
											  config->stage2.global_host->babysteps.n * sizeof(mp_strided_t)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &batch->host[offset + stream]->babysteps.y_tmp,
											  config->stage2.global_host->babysteps.n * sizeof(mp_strided_t)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &batch->host[offset + stream]->babysteps.z,
											  config->stage2.global_host->babysteps.n * sizeof(mp_strided_t)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &batch->host[offset + stream]->babysteps.t,
											  config->stage2.global_host->babysteps.n * sizeof(mp_strided_t)));
		}

		size_t mem_free, mem_total;
		cudaMemGetInfo(&mem_free, &mem_total); // in bytes

		// Use all remaining memory for point buffer in stage 2 (except 100 MB)
		size_t per_thread = sizeof(mp_strided_t) * 3 * config->n_cuda_streams;
		size_t bufsize_giantsteps = min((mem_free - (100 * 1000 * 1000)) / per_thread,
										config->stage2.global_host->giantsteps_n);
		LOG_DEBUG("[Device %i] Stage 2 init memfree minus: %zu", dev, (mem_free - (10 * 1000 * 1000)));
		LOG_DEBUG("[Device %i] Stage 2 init per thread: %zuB", dev, per_thread);
		LOG_DEBUG("[Device %i] Stage 2 init gs points: %d", dev, config->stage2.global_host->giantsteps_n);
		LOG_INFO("[Device %i] Stage 2 Giantstep buffer: %zuB free memory, using %zuB (%i points) per thread)",
				 dev,
				 mem_free,
				 per_thread,
				 bufsize_giantsteps);

		for (int stream = 0; stream < config->n_cuda_streams; stream++) {
			batch->host[offset + stream]->giantsteps.bufsize = bufsize_giantsteps;

			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &batch->host[offset + stream]->giantsteps.y,
											  bufsize_giantsteps * sizeof(mp_strided_t)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &batch->host[offset + stream]->giantsteps.z,
											  bufsize_giantsteps * sizeof(mp_strided_t)));
			CUDA_SAFE_CALL_NO_SYNC(cudaMalloc((void **) &batch->host[offset + stream]->giantsteps.t,
											  bufsize_giantsteps * sizeof(mp_strided_t)));

		}
	}
	LOG_INFO("Stage 2 init done");
}

void ecm_stage2(run_config config, batch_naf *batch, size_t stream) {

	LOG_DEBUG("[Thread %d] Starting ECM Stage 2...", stream);

	if(batch->host[stream]->n_jobs == 0){
		LOG_WARNING("[Thread %d] No tasks in batch for Stage 2", stream);
		return;
	}
	LOG_DEBUG("[Thread %d] %d tasks in batch for Stage 2", stream, batch->host[stream]->n_jobs);

	LOG_VERBOSE("[CUDA Stream %p] Launching stage2 exec kernel.", config->cuda_streams[stream]);
  	cuda_tw_ed_stage2<<<batch->host[stream]->cuda_blocks, config->cuda_threads_per_block,
  		sizeof(shared_mem_cache)*config->cuda_threads_per_block, config->cuda_streams[stream]>>>
			(&batch->dev[stream]->job, config->dev_ctx[batch->host[stream]->device].stage2.global_dev,
					batch->host[stream]->babysteps.y, batch->host[stream]->babysteps.y_tmp, batch->host[stream]->babysteps.z, batch->host[stream]->babysteps.t,
					batch->host[stream]->giantsteps.y, batch->host[stream]->giantsteps.z, batch->host[stream]->giantsteps.t, batch->host[stream]->giantsteps.bufsize);

	/* Copy batch from device to host */
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpyAsync(&batch->host[stream]->job,
										   &batch->dev[stream]->job,
										   sizeof(batch_job_data_naf),
										   cudaMemcpyDeviceToHost,
										   config->cuda_streams[stream]));

  	cudaStreamSynchronize(config->cuda_streams[stream]);

	batch_finished_cb_stage2(batch->host[stream]);

  	return;
}

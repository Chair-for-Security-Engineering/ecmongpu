#include <stdlib.h>
#include <getopt.h>
#include <gmp.h>
#include <omp.h>
#include <unistd.h>
#include <ecm/batch.h>
#include <cuda_runtime.h>
#include <cudautil.h>
#include <sys/stat.h>
#include <errno.h>
#include <libgen.h>
#include <errno.h>

#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "ecc/twisted_edwards.h"
#include "ecm/twisted_edwards.h"
#include "ecm/batch.h"
#include "ecm/factor_task.h"
#include "config/config.h"
#include "ecm/ecm.h"
#include "ecc/naf.h"

#include "log.h"
#include "ecm/stage1.h"
#include "ecm/batch.h"

__constant__ naf_limb const_scalar[60000 / sizeof(naf_limb)];


void ecm_stage1_init(run_config config) {
	/* Compute scalar and NAF form or read chain from file */
	if (config->stage1.host_bound_naf == NULL) {

		if (config->b1chain != NULL) {
			char chainpath[1000];
			ssize_t ret = readlink("/proc/self/exe", chainpath, 1000);
			if (ret == -1) {
				LOG_FATAL("Could not find chain directory.");
				exit(EXIT_FAILURE);
			}
			dirname(chainpath);
			strcat(chainpath, "/res/");
			strcat(chainpath, config->b1chain);

			LOG_DEBUG("Reading Stage1 chain from %s", chainpath);

			FILE *chainfile = fopen(chainpath, "rb");
			if (chainfile) {
				struct stat st;
				fstat(fileno(chainfile), &st);
				config->stage1.host_bound_naf = (naf_t) malloc(st.st_size * sizeof(naf_limb));
				config->stage1.bound_naf_digits = fread(config->stage1.host_bound_naf, 1, st.st_size, chainfile);
				LOG_WARNING("read %d chain bytes", config->stage1.bound_naf_digits);
			} else {
				LOG_FATAL("error (%d) opening chain file: %s\n", errno, chainpath);
				exit(EXIT_FAILURE);
			}
		} else {
			LOG_DEBUG("Precomputing Stage1 scalar and NAF form");
			mpz_t gmp_scalar;
			mpz_init(gmp_scalar);

			if (config->powersmooth == true) {
				LOG_DEBUG("Using powersmoothness bound B1");
				mpz_set_ui(gmp_scalar, 1);
				for (int i = 2; i <= config->b1; i++) {
					mpz_lcm_ui(gmp_scalar, gmp_scalar, i);
				}
			} else {
				LOG_DEBUG("Using smoothness bound B1");
				mpz_primorial_ui(gmp_scalar, config->b1);
			}
			size_t naf_size = mpz_sizeinbase(gmp_scalar, 2) + 2;
			config->stage1.host_bound_naf = (naf_t) malloc(naf_size * sizeof(naf_limb));
			config->stage1.bound_naf_digits = to_naf(config->stage1.host_bound_naf,
													 naf_size,
													 gmp_scalar,
													 NAF_WINDOW_SIZE);
			if (config->stage1.bound_naf_digits < 0) {
				LOG_FATAL("scalar too big for NAF form");
				exit(EXIT_FAILURE);
			}
			mpz_clear(gmp_scalar);
		}
	}

	// Initialize all devices
	for (int dev = 0; dev < config->devices; dev++) {
		cudaSetDevice(dev);
		if (config->dev_ctx[dev].stage1.dev_bound_naf == NULL) {
			LOG_INFO("Stage 1 Initialization of device #%i", dev);
			CUDA_SAFE_CALL(cudaMalloc((void **) &config->dev_ctx[dev].stage1.dev_bound_naf,
									  config->stage1.bound_naf_digits * sizeof(naf_limb)));
			CUDA_SAFE_CALL(cudaMemcpy(config->dev_ctx[dev].stage1.dev_bound_naf,
									  config->stage1.host_bound_naf,
									  config->stage1.bound_naf_digits * sizeof(naf_limb),
									  cudaMemcpyHostToDevice));
		}
	}
}


void ecm_stage1(run_config config, batch_naf *batch, size_t stream) {


	batch->host[stream]->cuda_blocks =
			(batch->host[stream]->n_jobs + config->cuda_threads_per_block - 1) / config->cuda_threads_per_block;


	/* Copy batch from host to device */
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpyAsync(&batch->dev[stream]->job,
										   &batch->host[stream]->job,
										   sizeof(batch_job_data_naf),
										   cudaMemcpyHostToDevice,
										   config->cuda_streams[stream]));

	/* Launch multiplication */
	LOG_DEBUG("[Device %d] [CUDA Stream %p] Launching scalarmult kernel.",
			  batch->host[stream]->device,
			  config->cuda_streams[stream]);
	cuda_tw_ed_smul_naf_batch<<<batch->host[stream]->cuda_blocks, config->cuda_threads_per_block,
			(sizeof(shared_mem_cache)) * config->cuda_threads_per_block, config->cuda_streams[stream]>>>
					(&batch->dev[stream]->job,
					config->dev_ctx[batch->host[stream]->device].stage1.dev_bound_naf,
					config->stage1.bound_naf_digits);

	/* Copy batch from device to host */
	CUDA_SAFE_CALL_NO_SYNC(cudaMemcpyAsync(&batch->host[stream]->job,
										   &batch->dev[stream]->job,
										   sizeof(batch_job_data_naf),
										   cudaMemcpyDeviceToHost,
										   config->cuda_streams[stream]));

	cudaStreamSynchronize(config->cuda_streams[stream]);
	batch_finished_cb_stage1(batch->host[stream]);

}

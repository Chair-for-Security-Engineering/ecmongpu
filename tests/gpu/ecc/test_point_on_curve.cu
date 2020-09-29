#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "cudautil.h"
#include "mp/mp.h"
#include "log.h"
#include "gmp.h"
#include "mp/mp_montgomery.h"
#include "cuda.h"
#include "mp/gmp_conversion.h"
#include "ecc/twisted_edwards.h"
#include "ecm/batch.h"
#include "ecm/ecm.h"
#include "test/testutil.h"

__host__
int test() {

	int cuda_threads_per_block = BATCH_JOB_SIZE / BLOCK_SIZE;
	int cuda_blocks = BATCH_JOB_SIZE / cuda_threads_per_block;


	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	mpz_t gmp_n;
	mp_t n;

	mpz_init(gmp_n);
	/* n has to be an odd modulus */
	mpz_urandomb(gmp_n, gmprand, BITWIDTH);
	mpz_nextprime(gmp_n, gmp_n);
	//mpz_setbit(gmp_n, 0);

	mpz_to_mp(n, gmp_n);
	batch_job *batch = (batch_job *)malloc(sizeof(batch_job));
	batch_job *batch2 = (batch_job *)malloc(sizeof(batch_job));

	point_gkl2016 phelper;
  pthread_mutex_init(&phelper.mutex, NULL);
	mpz_init_set_ui(phelper.x, 5);
	mpz_init_set_ui(phelper.y, 8);

	for (int curve_gen = 0; curve_gen < job_generators_len; curve_gen++) {
		LOG_DEBUG("Using curve generator %d: %s", curve_gen, job_generators_names[curve_gen]);
		for (int i = 0; i < BATCH_JOB_SIZE; i++) {
			mon_info_populate(n, &batch->job[i].mon_info);
			(job_generators[curve_gen])(&batch->job[i].point, &batch->job[i].curve, &batch->job[i].mon_info, gmprand, (void*)&phelper);
			batch->job[i].on_curve = 0;
		}

		memcpy(batch2, batch, sizeof(batch_job));


		batch_job *dev_batch;
		cudaMalloc(&dev_batch, sizeof(batch_job));

#ifdef LOG_LEVEL_VERBOSE_ENABLED
		LOG_VERBOSE("point before:\n");
		tw_ed_print_point(&batch->job[0].point, &batch->job[0].mon_info);
#endif

		CUDA_SAFE_CALL(cudaMemcpy(dev_batch, batch, sizeof(batch_job), cudaMemcpyHostToDevice));
		cuda_tw_ed_point_on_curve<< < cuda_blocks, cuda_threads_per_block>> > (dev_batch->job);
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		CUDA_SAFE_CALL(cudaPeekAtLastError());
		CUDA_SAFE_CALL(cudaMemcpy(batch, dev_batch, sizeof(batch_job), cudaMemcpyDeviceToHost));

#ifdef LOG_LEVEL_VERBOSE_ENABLED
		LOG_VERBOSE("after:\n");
		tw_ed_print_point(&batch->job[0].point, &batch->job[0].mon_info);
#endif

		for (int i = 0; i < BATCH_JOB_SIZE; i++) {
			if (!batch->job[i].on_curve) {
				printf("[Job %d] Error: point should be on curve\n", i);
				TEST_FAILURE;
			}
		}
	}

	/* Cleanup GMP */
	mpz_clear(gmp_n);
	gmp_randclear(gmprand);

	TEST_SUCCESS;
}

TEST_MAIN;


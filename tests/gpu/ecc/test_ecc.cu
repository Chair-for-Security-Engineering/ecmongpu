#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <gmp.h>

#include "cudautil.h"
#include "mp/mp.h"
#include "log.h"
#include "mp/mp_montgomery.h"
#include "cuda.h"
#include "ecm/ecm.h"
#include "mp/gmp_conversion.h"
#include "ecc/twisted_edwards.h"
#include "ecm/batch.h"
#include "test/testutil.h"


__global__
void cuda_tw_ed_double_batch(batch_job *batch) {
	int myjob = (blockDim.x * blockIdx.x) + threadIdx.x;
	tw_ed_double(&batch->job[myjob].point, &batch->job[myjob].point, &batch->job[myjob].curve, &batch->job[myjob].mon_info, true);
}

int test() {
	int cuda_threads_per_block = BATCH_JOB_SIZE / BLOCK_SIZE;
	int cuda_blocks = BATCH_JOB_SIZE / cuda_threads_per_block;

	srand(time(NULL));

	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	mpz_t gmp_n;
	mp_t n;

	/* n is prime. all curve arithmetic should stay on curve */
	mpz_init(gmp_n);
	mpz_urandomb(gmp_n, gmprand, BITWIDTH);
	mpz_nextprime(gmp_n, gmp_n);

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
			(job_generators[curve_gen])(&batch->job[i].point, &batch->job[i].curve, &batch->job[i].mon_info, gmprand, (void *)&phelper);

			batch->job[i].on_curve = tw_ed_point_on_curve(&batch->job[i].point, &batch->job[i].curve, &batch->job[i].mon_info);
			if (!batch->job[i].on_curve) {
				printf("Error: point not on curve\n");
				TEST_FAILURE;
			}
		}

		memcpy(batch2, batch, sizeof(batch_job));


		batch_job *dev_batch;
		cudaMalloc(&dev_batch, sizeof(batch_job));

#ifdef LOG_LEVEL_VERBOSE_ENABLED
		LOG_VERBOSE("point before:\n");
		tw_ed_print_point(&batch->job[0].point, &batch->job[0].mon_info);
#endif

		CUDA_SAFE_CALL(cudaMemcpy(dev_batch, batch, sizeof(batch_job), cudaMemcpyHostToDevice));
		for(int i=0; i < 100; i++){
			cuda_tw_ed_double_batch << < cuda_blocks, cuda_threads_per_block >> > (dev_batch);
		}
		CUDA_SAFE_CALL(cudaDeviceSynchronize());
		CUDA_SAFE_CALL(cudaPeekAtLastError());
		CUDA_SAFE_CALL(cudaMemcpy(batch2, dev_batch, sizeof(batch_job), cudaMemcpyDeviceToHost));

#ifdef LOG_LEVEL_VERBOSE_ENABLED
		LOG_VERBOSE("after:\n");
		tw_ed_print_point(&batch2->job[0].point, &batch2->job[0].mon_info);
#endif

		if (memcmp(batch, batch2, sizeof(batch_job)) == 0) {
			// CUDA kernel did not run
			printf("Error: CUDA kernel did not run\n");
			TEST_FAILURE;
		}

		for (int i = 0; i < BATCH_JOB_SIZE; i++) {
			batch->job[i].on_curve = tw_ed_point_on_curve(&batch->job[i].point, &batch->job[i].curve, &batch->job[i].mon_info);
			if (!batch->job[i].on_curve) {
				printf("Error: point not on curve\n");
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


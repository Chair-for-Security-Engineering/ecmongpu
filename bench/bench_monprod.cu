#include <gmp.h>
#include <cuda.h>
#include "mp/mp.h"
#include "log.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"
#include <time.h>


#define ITERATIONS 1000000
#define BLOCKS 256
#define THREADS_PER_BLOCK (BATCH_JOB_SIZE/BLOCKS)



typedef struct __monprod_bench {
	mon_info_strided info;
	mp_strided_t a;
	mp_strided_t b;
} monprod_bench;

__global__
void bench(monprod_bench *data){
	int myjob = (blockDim.x * blockIdx.x) + threadIdx.x;

	mon_info info;
	mp_copy_cs(info.R2,  data->info.R2, myjob);
	mp_copy_cs(info.n,   data->info.n, myjob);
	info.mu = data->info.mu[myjob];

	mp_t a;
	mp_copy_cs(a,  data->a, myjob);

	mp_t b;
	mp_copy_cs(b,  data->b, myjob);

	for(size_t i = 0; i < ITERATIONS; i++){
		mon_prod(a, a, b, &info);
	}

	mp_copy_sc(data->a, myjob, a);
}

int test() {
	srand(time(NULL));
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	monprod_bench *data = (monprod_bench *)malloc(sizeof(monprod_bench));


	for (int job = 0; job < BATCH_JOB_SIZE; job++) {
		mpz_t gmp_a, gmp_b, gmp_n;

		mpz_init(gmp_a);
		mpz_init(gmp_b);
		mpz_init(gmp_n);

		mp_t a, b, n;

		mpz_urandomb(gmp_n, gmprand, BITWIDTH);
		/* n has to be an odd modulus */
		mpz_setbit(gmp_n, 0);
		mpz_urandomb(gmp_a, gmprand, BITWIDTH);
		mpz_urandomb(gmp_b, gmprand, BITWIDTH);

		// Reduce operands mod n (to be safe)
		mpz_mod(gmp_a, gmp_a, gmp_n);
		mpz_mod(gmp_b, gmp_b, gmp_n);

		mpz_to_mp(n, gmp_n);

		/* Allocate and calculate Montgomery Constants for this modulus */
		mon_info info;
		mon_info_populate(n, &info);

		/* Operands */
		mpz_to_mp(a, gmp_a);
		mpz_to_mp(b, gmp_b);

		/* Compute Montgomery Transform of b and b */
		to_mon(a, a, &info);
		to_mon(b, b, &info);

		/* Copy data to bench struct */
		mp_copy_sc(data->info.R2, job,  info.R2);
		mp_copy_sc(data->info.n, job,   info.n);
		data->info.mu[job] = info.mu;

		mp_copy_sc(data->a, job, a);
		mp_copy_sc(data->b, job, b);


		/* Cleanup GMP */
		mpz_clear(gmp_a);
		mpz_clear(gmp_b);
		mpz_clear(gmp_n);
	}


	monprod_bench *data_dev;
	CUDA_SAFE_CALL(cudaMalloc((void **)&data_dev, sizeof(monprod_bench)));
	CUDA_SAFE_CALL(cudaMemcpy(data_dev, data, sizeof(monprod_bench), cudaMemcpyHostToDevice));

      	struct timespec start={0,0}, end={0,0};
      	double tmp_time, m_per_sec;
        clock_gettime(CLOCK_MONOTONIC, &start);
	bench<<<BLOCKS, THREADS_PER_BLOCK>>>(data_dev);
	cudaDeviceSynchronize();
        clock_gettime(CLOCK_MONOTONIC, &end);

	CUDA_SAFE_CALL(cudaMemcpy(data, data_dev, sizeof(monprod_bench), cudaMemcpyDeviceToHost));

        tmp_time = (((double)end.tv_sec + 1.0e-9*end.tv_nsec) - ((double)start.tv_sec + 1.0e-9*start.tv_nsec));
	m_per_sec = ((double) ((double)ITERATIONS*BATCH_JOB_SIZE)) / (tmp_time);
#ifdef MON_PROD_CIOS 
	LOG_INFO("CIOS");
#endif
#ifdef MON_PROD_CIOS_XMAD
	LOG_INFO("CIOS_XMAD");
#endif
#ifdef MON_PROD_FIOS 
	LOG_INFO("FIOS");
#endif
#ifdef MON_PROD_FIPS 
	LOG_INFO("FIPS");
#endif
	LOG_INFO("BITWIDTH: %d", BITWIDTH);
	LOG_INFO("THREADS: %d", BATCH_JOB_SIZE);
	LOG_INFO("BLOCKS: %d", BLOCKS);
	LOG_INFO("THREADS_PER_BLOCK: %d", THREADS_PER_BLOCK);
	LOG_INFO("%.3f mult/s", m_per_sec);
	gmp_randclear(gmprand);
	TEST_SUCCESS;
}

TEST_MAIN;


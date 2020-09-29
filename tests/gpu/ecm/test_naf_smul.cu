#include <stdlib.h>
#include <time.h>

#include "mp/mp.h"
#include "log.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "ecc/twisted_edwards.h"
#include "ecc/naf.h"
#include "ecm/batch.h"
#include "cudautil.h"
#include "test/testutil.h"
#include "ecm/stage1.h"

#define DEBUG

int test() {
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	srand(time(NULL));
	gmp_randseed_ui(gmprand, rand());

	mpz_t gmp_n;
	mp_t n;

	/* n is prime. all curve arithmetic should stay on curve */
	mpz_init(gmp_n);
	mpz_urandomb(gmp_n, gmprand, BITWIDTH);
	mpz_nextprime(gmp_n, gmp_n);
	gmp_printf("gmp_n: %Zi\n", gmp_n);
	printf("gmp_n bits: %ld\n", mpz_sizeinbase(gmp_n,2));

	mpz_to_mp(n, gmp_n);

	/* Prepare batches */

	batch_job *batch = (batch_job *) malloc(sizeof(batch_job));
	batch_job *dev_batch;
	CUDA_SAFE_CALL(cudaMalloc(&dev_batch, sizeof(batch_job)));

	batch_job_naf *batch_naf = (batch_job_naf *) malloc(sizeof(batch_job_naf));
	batch_job_naf *dev_batch_naf;
	CUDA_SAFE_CALL(cudaMalloc(&dev_batch_naf, sizeof(batch_job_naf)));

	point_gkl2016 phelper;
  pthread_mutex_init(&phelper.mutex, NULL);
	mpz_init_set_ui(phelper.x, 5);
	mpz_init_set_ui(phelper.y, 8);

	for (int curve_gen = 0; curve_gen < job_generators_len; curve_gen++) {
		for (int i = 0; i < BATCH_JOB_SIZE; i++) {
			mon_info info_naf;
			mon_info_populate(n, &batch->job[i].mon_info);
			mon_info_populate(n, &info_naf);


	    		/* Copy mon info to strided version */
	    		mp_copy_sc(batch_naf->job.mon_info_strided.n, i, batch->job[i].mon_info.n);
	    		mp_copy_sc(batch_naf->job.mon_info_strided.R2, i, batch->job[i].mon_info.R2);
	    		batch_naf->job.mon_info_strided.mu[i] = batch->job[i].mon_info.mu;

			LOG_DEBUG("Using curve generator %d: %s", curve_gen, job_generators_names[curve_gen]);
			(job_generators[curve_gen])(&batch->job[i].point, &batch->job[i].curve, &batch->job[i].mon_info, gmprand, (void *)&phelper);

			tw_ed_copy_point_sc(&batch_naf->job.point_strided, i, &batch->job[i].point);

	    		/* Copy curve info to strided version */
	    		mp_copy_sc(batch_naf->job.curve_strided.d,  i, batch->job[i].curve.d);
	    		mp_copy_sc(batch_naf->job.curve_strided.k, i, batch->job[i].curve.k);

			if (!tw_ed_point_on_curve(&batch->job[i].point, &batch->job[i].curve, &batch->job[i].mon_info)) {
				printf("Point not on curve (reg)\n");
				TEST_FAILURE;
			}
		}
	}


#ifdef DEBUG
	/* print both jobs first point */
	tw_ed_print_point_strided(&batch_naf->job.point_strided, 0, &batch->job[0].mon_info);
	tw_ed_print_point(&batch->job[0].point, &batch->job[0].mon_info);
#endif



	/* Prepare scalar */

	mpz_t gmp_scalar;
	mpz_init_set_ui(gmp_scalar, 1);
	for(int i = 1; i <= 4096; i++){
	  mpz_lcm_ui(gmp_scalar, gmp_scalar, i);
	}
	//mpz_primorial_ui(gmp_scalar, 4096);


	/* Transfer naf scalar */
	int naf_size = mpz_sizeinbase(gmp_scalar, 2) + 2;
	printf("scalar_naf_digits: %d\n", naf_size);
	naf_t h_scalar_naf = (naf_t) malloc(naf_size * sizeof(int8_t));
	naf_t scalar_naf_dev;
	int scalar_naf_digits = to_naf(h_scalar_naf, naf_size, gmp_scalar, NAF_WINDOW_SIZE);
	printf("scalar_naf_digits: %d\n", scalar_naf_digits);
	if (scalar_naf_digits < 0) {
		printf("scalar too big for NAF form\n");
		TEST_FAILURE;
	}
	CUDA_SAFE_CALL(cudaMalloc((void **) &scalar_naf_dev, scalar_naf_digits * sizeof(int8_t)));

#ifdef DEBUG
	gmp_printf("NAF scalar: %Zi\n", gmp_scalar);

	printf("NAF: ");
	print_naf(h_scalar_naf, scalar_naf_digits);
#endif

	CUDA_SAFE_CALL(cudaMemcpy(scalar_naf_dev,
							  h_scalar_naf,
							  scalar_naf_digits * sizeof(int8_t),
							  cudaMemcpyHostToDevice));


#ifdef DEBUG
  int job = 1;
  LOG_WARNING("Point naf strided (before)")
  tw_ed_print_point_strided(&batch_naf->job.point_strided, job, &batch->job[job].mon_info);
  LOG_WARNING("Point reg (before)")
  tw_ed_print_point(&batch->job[job].point, &batch->job[job].mon_info);
#endif

	/* Compute NAF version */
	CUDA_SAFE_CALL(cudaMemcpy(dev_batch_naf, batch_naf, sizeof(batch_job_naf), cudaMemcpyHostToDevice));
	//tw_ed_naf_precompute<< < BATCH_JOB_SIZE / BLOCK_SIZE, BLOCK_SIZE, (sizeof(shared_mem_cache))*(BLOCK_SIZE)  >> > (&dev_batch_naf->job);
  	printf("shared mem: %zu\n", sizeof(shared_mem_cache)*(BLOCK_SIZE));
	cuda_tw_ed_smul_naf_batch << < BATCH_JOB_SIZE / BLOCK_SIZE, BLOCK_SIZE, (sizeof(shared_mem_cache))*(BLOCK_SIZE)  >> > (&dev_batch_naf->job, scalar_naf_dev, scalar_naf_digits);
	CUDA_SAFE_CALL(cudaMemcpy(batch_naf, dev_batch_naf, sizeof(batch_job_naf), cudaMemcpyDeviceToHost));

#ifdef DEBUG
  for(int p = 1; p <= NAF_MAX_PRECOMPUTED; p+=2){
    LOG_WARNING("Point precomp[%d] (naf)", p);
    tw_ed_print_point_strided(&batch_naf->job.precomputed_strided[__naf_to_index(p)], job, &batch->job[job].mon_info);
  }

  LOG_WARNING("Point naf strided (after)")
  tw_ed_print_point_strided(&batch_naf->job.point_strided, job, &batch->job[job].mon_info);
#endif


	/* Transfer regular scalar */
	unsigned int scalar_bitlength = mpz_sizeinbase(gmp_scalar, 2);
	size_t scalar_limbs = (scalar_bitlength + LIMB_BITS - 1) / LIMB_BITS;
	printf("scalar limbs: %zu", scalar_limbs);
	mp_p h_scalar = (mp_limb *) malloc(scalar_limbs * sizeof(mp_limb));
	mpz_to_mp_limbs(h_scalar, gmp_scalar, scalar_limbs);
	mp_p scalar_dev;
	CUDA_SAFE_CALL(cudaMalloc((void **) &scalar_dev, scalar_limbs * sizeof(mp_limb)));
	mp_copy_to_dev_limbs(scalar_dev, h_scalar, scalar_limbs);

	/* Compute Dbl&Add version */
	CUDA_SAFE_CALL(cudaMemcpy(dev_batch, batch, sizeof(batch_job), cudaMemcpyHostToDevice));
	cuda_tw_ed_smul_batch << < BATCH_JOB_SIZE / BLOCK_SIZE, BLOCK_SIZE >> > (dev_batch, scalar_dev, scalar_bitlength);
	CUDA_SAFE_CALL(cudaMemcpy(batch, dev_batch, sizeof(batch_job), cudaMemcpyDeviceToHost));

#ifdef DEBUG
  LOG_WARNING("Point reg (after)")
  tw_ed_print_point(&batch->job[job].point, &batch->job[job].mon_info);
#endif

	/* check both points */
	for (int i = 0; i < BATCH_JOB_SIZE; i++) {
  		point_tw_ed p_naf;
		tw_ed_copy_point_cs(&p_naf, &batch_naf->job.point_strided, i);

		if (!tw_ed_point_on_curve(&batch->job[i].point, &batch->job[i].curve, &batch->job[i].mon_info)) {
			printf("Point not on curve (reg)\n");
		  	tw_ed_print_point(&batch->job[i].point, &batch->job[i].mon_info);
			TEST_FAILURE;
		}

		/* Copy point to strided version */
		//tw_ed_copy_point_sc(&batch->job.point_strided, i, &batch->job.point[job]);

		if (!tw_ed_point_on_curve(&p_naf, &batch->job[i].curve, &batch->job[i].mon_info)) {
			printf("[Job %d] Point not on curve (naf)\n", i);
  			tw_ed_print_point_strided(&batch_naf->job.point_strided, i, &batch->job[i].mon_info);
			TEST_FAILURE;
		}


#ifdef LOG_LEVEL_VERBOSE_ENABLED
	  if(i == 0){
		  LOG_VERBOSE("Before scaling\n");
		  tw_ed_print_point(&batch->job[i].point, &batch->job[i].mon_info);
  		  tw_ed_print_point_strided(&batch_naf->job.point_strided, i, &batch->job[i].mon_info);

		  LOG_VERBOSE("After regular coords\n");
		  tw_ed_to_reg(&batch->job[i].point, &batch->job[i].mon_info);
      		  tw_ed_to_reg(&p_naf, &batch->job[i].mon_inf);
		  tw_ed_print_point(&batch->job[i].point, &batch->job[i].mon_info);
		  tw_ed_print_point(&p_naf, &batch->job[i].mon_inf);
    }
#endif
		tw_ed_scale_point(&batch->job[i].point, &batch->job[i].point, &batch->job[i].mon_info);
		tw_ed_scale_point(&p_naf, &p_naf, &batch->job[i].mon_info);

#ifdef LOG_LEVEL_VERBOSE_ENABLED
	  if(i == 0){
		  LOG_VERBOSE("After scaling\n");
		  tw_ed_print_point(&batch->job[i].point, &batch->job[i].mon_info);
  		  tw_ed_print_point_strided(&batch_naf->job.point_strided, i, &batch->job[i].mon_info);
	  	LOG_VERBOSE("Result:\n");
	  	LOG_VERBOSE("regular:\n");
	  	tw_ed_print_point(&batch->job[i].point, &batch->job[i].mon_info);
	  	LOG_VERBOSE("naf:\n");
	  	tw_ed_print_point(&p_naf, &batch->job[i].mon_info);
	  }
#endif


		/* Check */
		if (mp_cmp(batch->job[i].point.x, p_naf.x) != 0 || \
           mp_cmp(batch->job[i].point.y, p_naf.y) != 0 ){
			printf("[job %d] not equal computations\n", i);
			printf("regular:\n");
		  tw_ed_print_point(&batch->job[i].point, &batch->job[i].mon_info);
			printf("naf:\n");
		  tw_ed_print_point(&p_naf, &batch->job[i].mon_info);
			TEST_FAILURE;
		}
	}
	

	/* Cleanup GMP */
	mpz_clear(gmp_n);
	gmp_randclear(gmprand);
    	TEST_SUCCESS;
	
}

TEST_MAIN;


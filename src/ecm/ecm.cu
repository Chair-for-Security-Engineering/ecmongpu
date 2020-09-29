#include "ecm/ecm.h"
#include "mp/mp.h"
#include "ecc/twisted_edwards.h"
#include "log.h"
#include "mp/gmp_conversion.h"
#include "ecm/factor_task.h"
#include <cudautil.h>
#include <ecm/stage2.h>

#define DEBUG_STAGE2 0


__global__
void cuda_tw_ed_smul_naf_batch(batch_job_data_naf batch[BATCH_JOB_SIZE],
							   naf_t scalar, size_t naf_digits) {

	int myjob = (blockDim.x * blockIdx.x) + threadIdx.x;

	extern __shared__ shared_mem_cache cache[];
	mp_copy_cs(cache[threadIdx.x].mon_info.n, batch->mon_info_strided.n, myjob);
	mp_copy_cs(cache[threadIdx.x].mon_info.R2, batch->mon_info_strided.R2, myjob); // needed here for inverse in scaling
	cache[threadIdx.x].mon_info.mu = batch->mon_info_strided.mu[myjob];

	mp_copy_cs(cache[threadIdx.x].curve.k, batch->curve_strided.k, myjob);
	mp_copy_cs(cache[threadIdx.x].curve.d, batch->curve_strided.d, myjob);

	point_tw_ed q;

	size_t batch_offset = 0;
	while (batch_offset < naf_digits) {
		tw_ed_naf_precompute(cache, batch, myjob);
		/* Returns offset to last (zero) byte of chain */
		batch_offset += tw_ed_smul_naf(
				&q,
				&cache[threadIdx.x].curve,
				batch->precomputed_strided,
				&cache[threadIdx.x].mon_info,
				scalar + batch_offset,
				naf_digits,
				myjob);
		tw_ed_copy_point_sc(&batch->point_strided, myjob, &q);
		/* Increase to start of next chain */
		batch_offset++;
  }
}



__global__
void cuda_tw_ed_stage2(batch_job_data_naf *batch, stage2_global *globals,
					   mp_strided_t *babysteps_y, mp_strided_t *babysteps_y_tmp, mp_strided_t *babysteps_z,
					   mp_strided_t *babysteps_t,
					   mp_strided_t *giantsteps_y, mp_strided_t *giantsteps_z, mp_strided_t *giantsteps_t,
					   size_t bufsize_giantsteps) {

	int myjob = (blockDim.x * blockIdx.x) + threadIdx.x;

	extern __shared__ shared_mem_cache cache[];
	mp_copy_cs(cache[threadIdx.x].mon_info.n, batch->mon_info_strided.n, myjob);
	mp_copy_cs(cache[threadIdx.x].mon_info.R2, batch->mon_info_strided.R2, myjob); // needed here for inverse in scaling
	cache[threadIdx.x].mon_info.mu = batch->mon_info_strided.mu[myjob];

	mp_copy_cs(cache[threadIdx.x].curve.k, batch->curve_strided.k, myjob);
	mp_copy_cs(cache[threadIdx.x].curve.d, batch->curve_strided.d, myjob);

	/* Compute babystep points */

	tw_ed_naf_precompute(cache, batch, myjob);

	point_tw_ed babystep_tmp;
	for (int i = 0; i < globals->babysteps.n; i++) {
		tw_ed_smul_naf(&babystep_tmp, &cache[threadIdx.x].curve,
					   batch->precomputed_strided,
					   &cache[threadIdx.x].mon_info,
					   globals->babysteps.naf[i],
					   globals->babysteps.naf_size[i],
					   myjob);

		mp_copy_sc(babysteps_y[i], myjob, babystep_tmp.y);
		mp_copy_sc(babysteps_z[i], myjob, babystep_tmp.z);
	}

	// Compute Point wq = w*Q
	point_tw_ed wq;
	point_tw_ed vwQ;

	tw_ed_smul_naf(&wq, &cache[threadIdx.x].curve,
				   batch->precomputed_strided,
				   &cache[threadIdx.x].mon_info,
				   globals->w.naf,
				   globals->w.naf_size,
				   myjob);
	tw_ed_copy_point(&vwQ, &wq);
	tw_ed_optimize_precomp(&wq, &wq, &cache[threadIdx.x].curve, &cache[threadIdx.x].mon_info);


	int stepcount = 0;

	mp_t res;
	mp_set_ui(res, 1);

	mp_t s, tmp, tmp_z, tmp_y;

	/* Loop over partitions of giantsteps */
	for (int cur_giantstep = 0; cur_giantstep < globals->giantsteps_n; cur_giantstep += bufsize_giantsteps) {
		int n_giantsteps = min(globals->giantsteps_n - cur_giantstep, bufsize_giantsteps);
		if(DEBUG_STAGE2 && blockIdx.x == 0 && threadIdx.x == 0) 
			printf("\n== GS base %i\n== GS num %i\n", cur_giantstep, n_giantsteps);

		/* Compute giantstep points */
		mp_t test;
		mp_copy(test, vwQ.z);
		for (int g = 0; g < n_giantsteps; g++) {
			mp_copy_sc(giantsteps_y[g], myjob, vwQ.y);
			mp_copy_sc(giantsteps_z[g], myjob, vwQ.z);
			tw_ed_add_precomp(&vwQ, &vwQ, &wq, &cache[threadIdx.x].curve, &cache[threadIdx.x].mon_info, true);
		}


		/* Compute lists */

		mp_copy_cs(tmp, babysteps_z[globals->babysteps.n - 1], myjob);
		mp_copy_sc(babysteps_t[globals->babysteps.n - 2], myjob, tmp);


		for (int i = globals->babysteps.n - 2; i >= 1; i--) {
			mp_copy_cs(tmp_z, babysteps_z[i], myjob);
			mon_prod(tmp, tmp_z, tmp, &cache[threadIdx.x].mon_info);
			mp_copy_sc(babysteps_t[i - 1], myjob, tmp);
		}
    
		mp_copy_cs(tmp_z, babysteps_z[0], myjob);
		mon_prod(tmp, tmp_z, tmp, &cache[threadIdx.x].mon_info);
		mp_copy_sc(giantsteps_t[n_giantsteps - 1], myjob, tmp);

		for (int i = n_giantsteps - 1; i >= 1; i--) {
			mp_copy_cs(tmp_z, giantsteps_z[i], myjob);
			mon_prod(tmp, tmp_z, tmp, &cache[threadIdx.x].mon_info);
			mp_copy_sc(giantsteps_t[i - 1], myjob, tmp);
		}


		// Canonicalize giantsteps[0]
		mp_copy_cs(tmp_y, giantsteps_y[0], myjob);
		mon_prod(tmp_y, tmp_y, tmp, &cache[threadIdx.x].mon_info);
		mp_copy_sc(giantsteps_y[0], myjob, tmp_y);

		if(DEBUG_STAGE2 && blockIdx.x == 0 && threadIdx.x == 0){
			mp_copy_cs(tmp_z, giantsteps_z[0], myjob);
			mon_prod(tmp_z, tmp_z, tmp, &cache[threadIdx.x].mon_info);

			printf("GS %i\tZ\t", 0);
			mp_print(tmp_z);
		}


		mp_set_ui(s, 1);

		for (int i = 1; i < n_giantsteps; i++) {
			mp_copy_cs(tmp_z, giantsteps_z[i - 1], myjob);
			mp_copy_cs(tmp, giantsteps_t[i], myjob);
			mon_prod(s, s, tmp_z, &cache[threadIdx.x].mon_info);
			mon_prod(tmp, s, tmp, &cache[threadIdx.x].mon_info);
			//mp_copy_sc(giantsteps_t[i], myjob, tmp);
			
			// Canonicalize giantsteps
			mp_copy_cs(tmp_y, giantsteps_y[i], myjob);
			mon_prod(tmp_y, tmp_y, tmp, &cache[threadIdx.x].mon_info);
			mp_copy_sc(giantsteps_y[i], myjob, tmp_y);

			if(DEBUG_STAGE2 && blockIdx.x == 0 && threadIdx.x == 0){
				mp_copy_cs(tmp_z, giantsteps_z[i], myjob);
				mon_prod(tmp_z, tmp_z, tmp, &cache[threadIdx.x].mon_info);

				printf("GS %i\tZ\t", i);
				mp_print(tmp_z);
			}
		}

		mp_copy_cs(tmp_z, giantsteps_z[n_giantsteps - 1], myjob);
		mp_copy_cs(tmp, babysteps_t[0], myjob);
		mon_prod(s, s, tmp_z, &cache[threadIdx.x].mon_info);
		mon_prod(tmp, s, tmp, &cache[threadIdx.x].mon_info);
		//mp_copy_sc(babysteps_t[0], myjob, tmp);

		
		// Canonicalize babystep[0]
		mp_copy_cs(tmp_y, babysteps_y[0], myjob);
		mon_prod(tmp_y, tmp_y, tmp, &cache[threadIdx.x].mon_info);
		mp_copy_sc(babysteps_y_tmp[0], myjob, tmp_y);

		if(DEBUG_STAGE2 && blockIdx.x == 0 && threadIdx.x == 0){
			mp_copy_cs(tmp_z, babysteps_z[0], myjob);
			mon_prod(tmp_z, tmp_z, tmp, &cache[threadIdx.x].mon_info);
			printf("BS %i\tZ\t", 0);
			mp_print(tmp_z);
		}

		for (int i = 1; i < globals->babysteps.n-1; i++) {
			mp_copy_cs(tmp_z, babysteps_z[i - 1], myjob);
			mp_copy_cs(tmp, babysteps_t[i], myjob);

			mon_prod(s, s, tmp_z, &cache[threadIdx.x].mon_info);
			mon_prod(tmp, s, tmp, &cache[threadIdx.x].mon_info);
			//mp_copy_sc(babysteps_t[i], myjob, tmp);

			// Canonicalize babysteps
			mp_copy_cs(tmp_y, babysteps_y[i], myjob);
			mon_prod(tmp_y, tmp_y, tmp, &cache[threadIdx.x].mon_info);
			mp_copy_sc(babysteps_y_tmp[i], myjob, tmp_y);

			if(DEBUG_STAGE2 && blockIdx.x == 0 && threadIdx.x == 0){
				mp_copy_cs(tmp_z, babysteps_z[i], myjob);
				//printf("BS %i\tZ0\t", i);
				//mp_print(tmp_z);
				mon_prod(tmp_z, tmp_z, tmp, &cache[threadIdx.x].mon_info);
				printf("BS %i\tZ\t", i);
				mp_print(tmp_z);
				//printf("BS %i\tT\t", i);
				//mp_print(tmp_t);
			}
		}

		int i = globals->babysteps.n-1;

		mp_copy_cs(tmp_z, babysteps_z[i - 1], myjob);
		mon_prod(s, s, tmp_z, &cache[threadIdx.x].mon_info);
		//mp_copy_sc(babysteps_t[i], myjob, s);

		// Canonicalize babystep[n-1]
		mp_copy_cs(tmp_y, babysteps_y[i], myjob);
		mon_prod(tmp_y, tmp_y, s, &cache[threadIdx.x].mon_info);
		mp_copy_sc(babysteps_y_tmp[i], myjob, tmp_y);

		if(DEBUG_STAGE2 && blockIdx.x == 0 && threadIdx.x == 0){
			mp_copy_cs(tmp_z, babysteps_z[i], myjob);
			//printf("BS %i\tZ0\t", i);
			//mp_print(tmp_z);
			mon_prod(tmp_z, tmp_z, s, &cache[threadIdx.x].mon_info);
			printf("BS %i\tZ\t", i);
			mp_print(tmp_z);
			//printf("BS %i\tT\t", i);
			//mp_print(tmp_t);
		}
	

		for (int g = 0; g < n_giantsteps; g++) {
			for (int b = 0; b < globals->babysteps.n; b++) {
				/* Check whether this combination yields a prime */
				if (mp_test_bit(globals->is_prime, stepcount)) {
					mp_t tmp_g, tmp_b;
					mp_copy_cs(tmp_b, babysteps_y_tmp[b], myjob);
					mp_copy_cs(tmp_g, giantsteps_y[g], myjob);
					mp_sub_mod(tmp_g, tmp_b, tmp_g, cache[threadIdx.x].mon_info.n);
					mon_prod(res, res, tmp_g, &cache[threadIdx.x].mon_info);
				}
				stepcount++;
			}
		}
	}

	mp_copy(batch->stage2_result[myjob], res);
}

__global__
void cuda_tw_ed_smul_batch(batch_job *batch, const mp_p scalar, const unsigned int scalar_bitlength) {
	int myjob = (blockDim.x * blockIdx.x) + threadIdx.x;

	point_tw_ed res;
	tw_ed_copy_point(&res, &batch->job[myjob].point);
	for (int bit = scalar_bitlength - 2; bit >= 0; bit--) {
		tw_ed_double(&res, &res, &batch->job[myjob].curve, &batch->job[myjob].mon_info, true);
		if (mp_test_bit(scalar, bit)) {
			tw_ed_add(&res,
					  &res,
					  &batch->job[myjob].point,
					  &batch->job[myjob].curve,
					  &batch->job[myjob].mon_info,
					  true);
		}
	}

	tw_ed_copy_point(&batch->job[myjob].point, &res);
}

__global__
void cuda_tw_ed_point_on_curve(batch_job_data *batch) {
	int myjob = (blockDim.x * blockIdx.x) + threadIdx.x;
	batch[myjob].on_curve = tw_ed_point_on_curve(&batch[myjob].point,
												 &batch[myjob].curve,
												 &batch[myjob].mon_info);
}

__global__
void cuda_tw_ed_point_on_curve_naf(batch_job_data_naf *batch) {
	int myjob = (blockDim.x * blockIdx.x) + threadIdx.x;
	curve_tw_ed curve;
	mp_copy_cs(curve.k, batch->curve_strided.k, myjob);
	mp_copy_cs(curve.d, batch->curve_strided.d, myjob);

	mon_info info;
	mp_copy_cs(info.n, batch->mon_info_strided.n, myjob);
	mp_copy_cs(info.R2, batch->mon_info_strided.R2, myjob);
	info.mu = batch->mon_info_strided.mu[myjob];

	point_tw_ed p;
	tw_ed_copy_point_cs(&p, &batch->point_strided, myjob);
	batch->on_curve[myjob] = tw_ed_point_on_curve(&p, &curve, &info);
}


__device__
void tw_ed_naf_precompute(shared_mem_cache *cache, batch_job_data_naf *batch, int myjob) {

	point_tw_ed tmp;
	point_tw_ed tmp_opt;

	/* Copy precomp[1] = 1P */
	tw_ed_copy_point_cs(&tmp, &batch->point_strided, myjob);
	tw_ed_optimize_precomp(&tmp_opt, &tmp, &cache[threadIdx.x].curve, &cache[threadIdx.x].mon_info);
	/* Copy to strided array */
	tw_ed_copy_point_sc(&batch->precomputed_strided[__naf_to_index(1)], myjob, &tmp_opt);
	
#if (NAF_MAX_PRECOMPUTED > 1)

	point_tw_ed p2;

	/* Compute p2 = P+P */
	tw_ed_double_cs(&p2, &batch->point_strided, myjob, &cache[threadIdx.x].curve, &cache[threadIdx.x].mon_info);


	for (int i = 3; i <= NAF_MAX_PRECOMPUTED; i += 2) {
		tw_ed_add(&tmp,
				  &tmp,
				  &p2,
				  &cache[threadIdx.x].curve, &cache[threadIdx.x].mon_info, true);

		/* Scale point to z=1 and optimize */
		tw_ed_optimize_precomp(&tmp_opt, &tmp, &cache[threadIdx.x].curve, &cache[threadIdx.x].mon_info);

		/* Copy to strided array */
		tw_ed_copy_point_sc(&batch->precomputed_strided[__naf_to_index(i)], myjob, &tmp_opt);
	}
#endif
}


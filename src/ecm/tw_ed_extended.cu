#include <cuda.h>
#include <cuda_runtime.h>
#include "ecc/twisted_edwards.h"
#include "ecm/factor_task.h"
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "mp/mp_montgomery.h"

#ifdef __cplusplus
extern "C" {
#endif


pthread_mutex_t mutex_gmp_rand = PTHREAD_MUTEX_INITIALIZER;


#define TRPL_MASK 0x80
#define DBL_MASK 0x40
#define NEG_MASK 0x20
#define VALUE_MASK 0x1f

__host__ __device__

size_t tw_ed_smul_naf(point_tw_ed *p,
					  const curve_tw_ed *curve,
					  const point_tw_ed_strided precomputed_strided[NAF_N_PRECOMPUTED],
					  const mon_info *info,
					  const naf_t scalar,
					  const size_t scalar_size,
					  const size_t myjob) {


	mp_set_ui(p->x, 0);
	mp_set_ui(p->y, 1);
	mp_set_ui(p->z, 1);
	mp_set_ui(p->t, 0);

	bool finish = false;
	size_t i = 0;
	while (!finish) {
		finish = (scalar[i + 1] == 0x00);

		uint8_t digit_val = scalar[i] & VALUE_MASK;

		/* double or triple
		   only compute extended coordinate (p.t) if addition is following
		 */
		if (scalar[i] & DBL_MASK) {
			tw_ed_double(p, p, NULL, info, (digit_val != 0) | finish);
		}
		if (scalar[i] & TRPL_MASK) {
			tw_ed_triple(p, p, NULL, info, (digit_val != 0) | finish);
		}
		if (digit_val != 0) {
			point_tw_ed delta;
			tw_ed_copy_point_cs(&delta, &precomputed_strided[__naf_to_index(digit_val)], myjob);
			if (scalar[i] & NEG_MASK) {
				tw_ed_point_invert_precomp(&delta, info);
			}

			tw_ed_add_precomp(p, p, &delta, curve, info, finish);

		};
		i++;
	}
	return i;
}

#ifdef OPTIMIZE_PRECOMP
/* Set optimized precomputed coordinates.
  Reduces online subtractions, additions, and saves one mult
  Zp := 1
  Xp := (Y-X)
  Yp := (Y+X)
  Tp := k*T
*/
__host__ __device__
void tw_ed_optimize_precomp(point_tw_ed *r, const point_tw_ed *p, curve_tw_ed *curve, mon_info *info) {

	tw_ed_scale_point(r, p, info);

	mp_t tmpx;
	mp_sub_mod(tmpx, r->y, r->x, info->n);
	mp_add_mod(r->y, r->y, r->x, info->n);
	mp_copy(r->x, tmpx);
	mon_prod(r->t, r->t, curve->k, info);
}
#else 
__host__ __device__
void tw_ed_optimize_precomp(point_tw_ed *r, const point_tw_ed *p, curve_tw_ed *curve, mon_info *info) {
	tw_ed_copy_point(r, p);
}
#endif /* OPTIMIZE_PRECOMP */


__host__
void batch_finished_cb_stage2(batch_job_naf *batch) {
	LOG_DEBUG("Stage 2 callback");
	mpz_t gmp_m, gmp_factor;
	mpz_init(gmp_factor);
	mpz_init(gmp_m);

	mp_t m;

	for (size_t job = 0; job < batch->n_jobs; job++) {
		mon_info info;
		mp_copy_cs(info.n, batch->job.mon_info_strided.n, job);
		mp_copy_cs(info.R2, batch->job.mon_info_strided.R2, job);
		info.mu = batch->job.mon_info_strided.mu[job];

		factor_task_inc_effort(batch->tasks_id[job], batch->config);

		/* Inverse Montgomery transform of stage2 result */
		mp_copy(m, batch->job.stage2_result[job]);
		from_mon(m, m, &info);
		mp_to_mpz(gmp_m, m);

		/* Add potential factor (if non-trivial) */
		task_add_factor(batch->tasks_id[job], batch->config, gmp_factor);

		/* Check if done with task and finish */
		task_finish(batch->tasks_id[job], batch->config);
	}

	mpz_clear(gmp_factor);
	mpz_clear(gmp_m);
}


__host__
void batch_finished_cb_stage1(batch_job_naf *batch) {
	LOG_VERBOSE("Stage 1 callback");

	for (size_t job = 0; job < batch->n_jobs; job++) {
		if (!batch->config->stage2.enabled) {
			factor_task_inc_effort(batch->tasks_id[job], batch->config);
		}

		mon_info info;
		mp_copy_cs(info.n, batch->job.mon_info_strided.n, job);
		mp_copy_cs(info.R2, batch->job.mon_info_strided.R2, job);
		info.mu = batch->job.mon_info_strided.mu[job];

		/* Check if point off curve (or all points are checked) */
		if (batch->config->stage1.check_all || !batch->job.on_curve[job]) {
			mp_t m;
			mp_copy_cs(m, batch->job.point_strided.x, job);
			task_add_factor_mp_mon(batch->tasks_id[job], batch->config, m, &info);

			mp_copy_cs(m, batch->job.point_strided.z, job);
			task_add_factor_mp_mon(batch->tasks_id[job], batch->config, m, &info);

			/* Check if done with task and finish */
			task_finish(batch->tasks_id[job], batch->config);
		}

	}
}


#ifdef __cplusplus
}
#endif

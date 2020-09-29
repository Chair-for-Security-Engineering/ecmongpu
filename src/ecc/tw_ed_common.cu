
#include <cuda.h>
#include <cuda_runtime.h>
#include "ecc/twisted_edwards.h"
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "ecc/twisted_edwards.h"
#include "mp/mp_montgomery.h"


#ifdef __cplusplus
extern "C" {
#endif

__host__ __device__
int tw_ed_double_cs(point_tw_ed *r, const point_tw_ed_strided *op, const size_t op_elem, const curve_tw_ed *curve,
					const mon_info *info) {
	point_tw_ed op_c;
	tw_ed_copy_point_cs(&op_c, op, op_elem);
	return tw_ed_double(r, &op_c, curve, info, true);
}


/* Binary left to right */
__host__ __device__
int tw_ed_smul(point_tw_ed *r, const point_tw_ed *op, const curve_tw_ed *curve, const mp_p scalar,
			   const unsigned int scalar_bitlength, const mon_info *info) {
	point_tw_ed res;
	for (int bit = scalar_bitlength - 1; bit >= 0; bit--) {
		tw_ed_double(&res, &res, curve, info, true);
		if (mp_test_bit(scalar, bit)) {
			tw_ed_add(&res, &res, op, curve, info, false);
		}
	}
	tw_ed_copy_point(r, &res);

	return 0;
}


__host__ __device__
int tw_ed_sub(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
			  const mon_info *info, bool extend) {
	/* Copy op2 to temporary point and invert */
	point_tw_ed nop2;
	tw_ed_copy_point(&nop2, op2);
	tw_ed_point_invert(&nop2, info);
	return tw_ed_add(r, op1, &nop2, curve, info, extend);
}


int tw_ed_init_curve(curve_tw_ed *curve, mp_t a, mp_t d, mon_info *info) {
	mp_copy(curve->d, d);
	curve->info = info;
	mp_add_mod(curve->k, d, d, info->n);
	return 0;
}

__host__
curve_tw_ed *tw_ed_copy_curve_to_dev(const curve_tw_ed *curve) {
	curve_tw_ed *dev_curve;
	cudaMalloc((void **) (&dev_curve), sizeof(curve_tw_ed));
	cudaMemcpy((void *) (dev_curve), curve, sizeof(curve_tw_ed), cudaMemcpyHostToDevice);
	return dev_curve;
}

__host__
int tw_ed_copy_point_to_dev(const point_tw_ed *dev_point, const point_tw_ed host_point, curve_tw_ed *dev_curve) {
	cudaMemcpy((void *) dev_point, &host_point, sizeof(point_tw_ed), cudaMemcpyHostToDevice);
	return 0;
}

int tw_ed_copy_point_from_dev(point_tw_ed *host_point, point_tw_ed *dev_point) {
	cudaMemcpy(host_point, dev_point, sizeof(point_tw_ed), cudaMemcpyDeviceToHost);
	return 0;
}

__host__ __device__
int tw_ed_copy_curve(curve_tw_ed *dest, const curve_tw_ed *src) {
	mp_copy(dest->d, src->d);
	mp_copy(dest->k, src->k);
	dest->info = src->info;
	return 0;
}

void tw_ed_print_point_strided(const point_tw_ed_strided *ps, size_t job, const mon_info *info) {
	point_tw_ed p;
	tw_ed_copy_point_cs(&p, ps, job);
	tw_ed_print_point(&p, info);
}


__host__ __device__
int tw_ed_point_on_curve(const point_tw_ed *p, const curve_tw_ed *curve, const mon_info *info) {
	/* Calculate -x² + y² = 1 + dx²y²
	 * 			 -(x * z^(-1))² + (y * z^(-1)² = 1 + d * (x * z^(-1))² * (y * z^(-1))²
	 * 			 y² * z^(-2) - x² * z^(-2) = 1 + d * y² * z^(-2) * x² * z^(-2)
	 * 			 z^(-2) * (y² - x²) = 1 + d * y² * x² * z^(-4) 						| * z²
	 * 			 z²(y² - x²) = z^(4) + d * y² * x² 								    | * z²
	 * 			 y²z² - x²z² - z^(4)  = d * y² * x² 								| - z^(4)
	 * 			 y²z² - x²z² - z^(4)  = d * y² * x² 								|
	 * 			 z²(y² - x² - z²)  = d * y² * x² 									|
	 */

	/* Transform to extended projective */
	point_tw_ed q;
	tw_ed_copy_point(&q, p);
	tw_ed_to_reg(&q, info);

	mp_t zsq, xsq, ysq, lhs, rhs;
	mon_prod(zsq, q.z, q.z, info);
	mon_prod(xsq, q.x, q.x, info);
	mon_prod(ysq, q.y, q.y, info);

	mon_prod(rhs, xsq, ysq, info);
	mon_prod(rhs, rhs, curve->d, info);

	mp_copy(lhs, ysq);
	mp_sub_mod(lhs, lhs, xsq, info->n);
	mp_sub_mod(lhs, lhs, zsq, info->n);
	mon_prod(lhs, lhs, zsq, info);

	return (mp_cmp(lhs, rhs) == 0);
}


job_generator const job_generators[] = {
		(job_generator) (&tw_ed_random_curve_naive),
		(job_generator) (&tw_ed_random_curve_gkl2016_j1),
		(job_generator) (&tw_ed_random_curve_gkl2016_j4)
};

/**
 * Names for Curve generation algorithms in job_generators. Shown in help message and log.
 */
const char *const job_generators_names[] = {
		"Naive",
		"GKL2016_j1",
		"GKL2016_j4"
};

/**
 * Length of the job_generators function pointer array.
 */
const int job_generators_len = 3;

#ifdef __cplusplus
}
#endif

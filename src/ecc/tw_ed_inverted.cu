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

/**
 * See http://hyperelliptic.org/EFD/g1p/auto-twisted-inverted.html#doubling-dbl-2008-bbjlp
 *
      A = X1^2
      B = Y1^2
      U = a*B
      C = A+U
      D = A-U
      E = (X1+Y1)^2-A-B
      X3 = C*D
      Y3 = E*(C-k*Z1^2)
      Z3 = D*E
 */
__host__ __device__
int tw_ed_double(point_tw_ed *r, const point_tw_ed *op, const curve_tw_ed *curve, const mon_info *info, bool extend) {

	mp_t a, b, u, c, d, e;

	//  A = X^2
	mon_square(a, op->x, info);

	//  B = Y^2
	mon_square(b, op->y, info);

	//  U = a*B = -1 * B = n - B
	mp_sub(u, info->n, b);

	//  C = A + U
	mp_add_mod(c, a, u, info->n);

	//  D = A - U
	mp_sub_mod(d, a, u, info->n);

	//  E = (X+Y)^2-A-B
	mp_add_mod(e, op->x, op->y, info->n);
	mon_square(e, e, info);
	mp_sub_mod(e, e, a, info->n);
	mp_sub_mod(e, e, b, info->n);

	//  X3 = C*D
	mon_prod_distinct(r->x, c, d, info);

	//  Y3 = E*(C - k*Z^2)
	mon_square(r->y, op->z, info);
	mon_prod(r->y, r->y, curve->k, info);

	// mp_sub_mod
	if (!mp_gt(c, r->y)) {
		mp_add(c, c, info->n);
	}
	mp_sub(r->y, c, r->y);

	mon_prod(r->y, r->y, e, info);

	//  Z3 = D*E
	mon_prod_distinct(r->z, d, e, info);

	return 0;
}


void tw_ed_point_invert(point_tw_ed *p, const mon_info *info) {
	mp_sub(p->x, info->n, p->x);
}




/** See http://hyperelliptic.org/EFD/g1p/auto-twisted-inverted.html#addition-add-2008-bbjlp
 *
 *  Strongly unified: Also correct for P1 == P2: P3 = P1 + P2 = P1*2
 *

      A = Z1*Z2
      B = d*A^2
      C = X1*X2
      D = Y1*Y2
      E = C*D
      H = C-a*D
      I = (X1+Y1)*(X2+Y2)-C-D
      X3 = (E+B)*H
      Y3 = (E-B)*I
      Z3 = A*H*I
 */
__host__ __device__
int tw_ed_add(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
			  const mon_info *info, bool extend) {

	mp_t a, b, c, d, e, h, i;
	mp_t tmp;

	//    A = Z1*Z2
	mon_prod_distinct(a, op1->z, op2->z, info);

	//    B = d*A^2
	mon_prod_distinct(b, a, a, info);
	mon_prod(b, b, curve->d, info);


	//    C = X1*X2
	mon_prod_distinct(c, op1->x, op2->x, info);

	//    D = Y1*Y2
	mon_prod_distinct(d, op1->y, op2->y, info);

	//    E = C*D
	mon_prod_distinct(e, c, d, info);

	//    H = C-a*D = C- (-1 * D) = C+D
	mp_add_mod(h, c, d, info->n);

	//    I = (X1+Y1) * (X2+Y2) - C - D
	mp_add_mod(i, op1->x, op1->y, info->n);
	mp_add_mod(tmp, op2->x, op2->y, info->n);
	mon_prod(i, i, tmp, info);
	mp_sub_mod(i, i, c, info->n);
	mp_sub_mod(i, i, d, info->n);

	//    X3 = (E+B)*H
	mp_add_mod(r->x, e, b, info->n);
	mon_prod(r->x, r->x, h, info);

	//    Y3 = (E-B)*I
	mp_sub_mod(r->y, e, b, info->n);
	mon_prod(r->y, r->y, i, info);

	//    Z3 = A*H*I
	mon_prod_distinct(r->z, h, i, info);
	mon_prod(r->z, r->z, a, info);

	return 0;
}


/**
 * Dummy implementation using repeated addition. No optimized tripling for inverted coordinates is implemented.
 */
int tw_ed_triple(point_tw_ed *r, const point_tw_ed *op, const curve_tw_ed *curve, const mon_info *info, bool extend) {
	point_tw_ed tmp;
	tw_ed_add(&tmp, op, op, curve, info, true);
	tw_ed_add(r, &tmp, op, curve, info, true);
	return 0;
}


#ifdef OPTIMIZE_PRECOMP
/** See http://hyperelliptic.org/EFD/g1p/auto-twisted-inverted.html#addition-madd-2008-bbjlp
 *
 *  Strongly unified: Also correct for P1 == P2: P3 = P1 + P2 = P1*2
 *

      B = d*Z1^2
      C = X1*X2
      D = Y1*Y2
      E = C*D
      H = C-a*D
      I = (X1+Y1)*(X2+Y2)-C-D
      X3 = (E+B)*H
      Y3 = (E-B)*I
      Z3 = Z1*H*I
 */
__host__ __device__
int tw_ed_add_precomp(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
					  const mon_info *info, bool extend) {

	mp_t b, c, d, e, h, i;
	mp_t tmp;

	//    B = d*Z1^2
	mon_prod_distinct(b, op1->z, op1->z, info);
	mon_prod(b, b, curve->d, info);

	//    C = X1*X2
	mon_prod_distinct(c, op1->x, op2->x, info);

	//    D = Y1*Y2
	mon_prod_distinct(d, op1->y, op2->y, info);

	//    E = C*D
	mon_prod_distinct(e, c, d, info);

	//    H = C-a*D = C- (-1 * D) = C+D
	mp_add_mod(h, c, d, info->n);

	//    I = (X1+Y1) * (X2+Y2) - C - D
	mp_add_mod(i, op1->x, op1->y, info->n);
	mp_add_mod(tmp, op2->x, op2->y, info->n);
	mon_prod(i, i, tmp, info);
	mp_sub_mod(i, i, c, info->n);
	mp_sub_mod(i, i, d, info->n);

	//    X3 = (E+B)*H
	mp_add_mod(r->x, e, b, info->n);
	mon_prod(r->x, r->x, h, info);

	//    Y3 = (E-B)*I
	mp_sub_mod(r->y, e, b, info->n);
	mon_prod(r->y, r->y, i, info);

	//    Z3 = Z1*H*I
	mon_prod_distinct(tmp, h, i, info);
	mon_prod(r->z, tmp, op1->z, info);

	return 0;
}
#else
__inline__
__host__ __device__
int tw_ed_add_precomp(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
					  const mon_info *info, bool extend) {
 return tw_ed_add(r, op1, op2, curve, info, extend);
}
#endif /* OPTIMIZE_PRECOMP */


void tw_ed_print_curve(const curve_tw_ed *c) {
	printf("Curve:\n");
	printf("  Montgomery:\n");
	mp_printf("\td:  %Zi\n", c->d);
	mp_printf("\tk: %Zi\n", c->k);

	mp_t tmp;

	printf("  Normal:\n");
	from_mon(tmp, c->d, c->info);
	mp_printf("\td:  %Zi\n", tmp);
	from_mon(tmp, c->k, c->info);
	mp_printf("\tk: %Zi\n", tmp);
}


__host__ __device__
int tw_ed_copy_point(point_tw_ed *dest, const point_tw_ed *src) {
	mp_copy(dest->x, src->x);
	mp_copy(dest->y, src->y);
	mp_copy(dest->z, src->z);
	return 0;
}

__host__ __device__
int tw_ed_copy_point_sc(point_tw_ed_strided *dest, const size_t dest_elem,
						const point_tw_ed *src) {
	mp_copy_sc(dest->x, dest_elem, src->x);
	mp_copy_sc(dest->y, dest_elem, src->y);
	mp_copy_sc(dest->z, dest_elem, src->z);
	return 0;
}

__host__ __device__
int tw_ed_copy_point_cs(point_tw_ed *dest,
						const point_tw_ed_strided *src, const size_t src_elem) {
	mp_copy_cs(dest->x, src->x, src_elem);
	mp_copy_cs(dest->y, src->y, src_elem);
	mp_copy_cs(dest->z, src->z, src_elem);
	return 0;
}

__host__ __device__
int tw_ed_copy_point_ss(point_tw_ed_strided *dest, const size_t dest_elem,
						const point_tw_ed_strided *src, const size_t src_elem) {
	mp_copy_ss(dest->x, dest_elem, src->x, src_elem);
	mp_copy_ss(dest->y, dest_elem, src->y, src_elem);
	mp_copy_ss(dest->z, dest_elem, src->z, src_elem);
	return 0;
}


void tw_ed_print_point(const point_tw_ed *p, const mon_info *info) {
	printf("Point:\n");
	printf("  Montgomery:\n");
	mp_printf("\tx: %Zi\n", p->x);
	mp_printf("\ty: %Zi\n", p->y);
	mp_printf("\tz: %Zi\n", p->z);

	mp_t tmp;
	printf("  Normal:\n");
	from_mon(tmp, p->x, info);
	mp_printf("\tx: %Zi\n", tmp);
	from_mon(tmp, p->y, info);
	mp_printf("\ty: %Zi\n", tmp);
	from_mon(tmp, p->z, info);
	mp_printf("\tz: %Zi\n", tmp);
}


__host__
__device__
void tw_ed_to_reg(point_tw_ed *p, const mon_info *info) {
	mp_t tmpx, tmpy;
	mp_copy(tmpx, p->x);
	mp_copy(tmpy, p->y);
	mon_prod(p->x, tmpy, p->z, info);
	mon_prod(p->y, tmpx, p->z, info);
	mon_prod(p->z, tmpx, tmpy, info);
}


bool tw_ed_scale_point(point_tw_ed *r, const point_tw_ed *p, const mon_info *info) {

	mp_t zinv;
	mon_inv(zinv, p->z, info);

	mon_prod(r->x, p->x, zinv, info);
	mon_prod(r->y, p->y, zinv, info);
	mon_prod(r->z, p->z, zinv, info);

	return true;
}

void tw_ed_random_curve_naive(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand) {

	/* Curve parameters */
	mpz_t gmp_a, gmp_d, gmp_n;
	/* Point (extended projective coordinates) */
	mpz_t gmp_x, gmp_y, gmp_z;//, gmp_t;

	mpz_inits(gmp_a, gmp_d, gmp_n, gmp_x, gmp_y, gmp_z, NULL);

	mp_t x, y, z, a, d;//, t;

	mp_to_mpz(gmp_n, info->n);

	pthread_mutex_lock(&mutex_gmp_rand);
	mpz_urandomb(gmp_x, gmprand, BITWIDTH);
	mpz_urandomb(gmp_y, gmprand, BITWIDTH);
	pthread_mutex_unlock(&mutex_gmp_rand);


	// Reduce coordinates mod n (to be safe)
	mpz_mod(gmp_x, gmp_x, gmp_n);
	mpz_mod(gmp_y, gmp_y, gmp_n);

	/* Calculate d = (-x² + y² - 1)/(x²y²) */
	mpz_t gmp_x2, gmp_y2, gmp_tmp;
	mpz_inits(gmp_tmp, gmp_x2, gmp_y2, NULL);

	mpz_mul(gmp_x2, gmp_x, gmp_x);
	mpz_mod(gmp_x2, gmp_x2, gmp_n);
	mpz_mul(gmp_y2, gmp_y, gmp_y);
	mpz_mod(gmp_y2, gmp_y2, gmp_n);

	mpz_sub(gmp_d, gmp_y2, gmp_x2);
	mpz_sub_ui(gmp_d, gmp_d, 1);
	mpz_mod(gmp_d, gmp_d, gmp_n);

	mpz_mul(gmp_tmp, gmp_x2, gmp_y2);
	mpz_invert(gmp_tmp, gmp_tmp, gmp_n);
	mpz_mul(gmp_d, gmp_d, gmp_tmp);
	mpz_mod(gmp_d, gmp_d, gmp_n);


	/* Transform curve parameters to Montgomery Form */
	mpz_set_si(gmp_a, -1);
	mpz_mod(gmp_a, gmp_a, gmp_n);
	mpz_to_mp(a, gmp_a);
	to_mon(a, a, info);
	mpz_to_mp(d, gmp_d);
	to_mon(d, d, info);

	/* Initialize curve with parameters */
	tw_ed_init_curve(curve, a, d, info);


	// x := y*z = y
	mpz_to_mp(x, gmp_y);
	to_mon(x, x, curve->info);
	mp_copy(p1->x, x);

	// y := x*z = x
	mpz_to_mp(y, gmp_x);
	to_mon(y, y, curve->info);
	mp_copy(p1->y, y);

	// z := x*y
	mpz_mul(gmp_z, gmp_x, gmp_y);
	mpz_mod(gmp_z, gmp_z, gmp_n);
	mpz_to_mp(z, gmp_z);
	to_mon(z, z, curve->info);
	mp_copy(p1->z, z);


	mpz_clear(gmp_a);
	mpz_clear(gmp_d);
	mpz_clear(gmp_n);
	mpz_clear(gmp_x);
	mpz_clear(gmp_y);
	mpz_clear(gmp_z);
	mpz_clear(gmp_tmp);
	mpz_clear(gmp_x2);
	mpz_clear(gmp_y2);
}


void tw_ed_random_curve_gkl2016_j1(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand) {

	/* Curve construction parameters */
	mpz_t gmp_ct, gmp_ct2, gmp_ct4, gmp_ct6, gmp_ce;

	/* Curve parameters */
	mpz_t gmp_a, gmp_d, gmp_n;
	/* Point (extended projective coordinates) */
	mpz_t gmp_x, gmp_y, gmp_z;//, gmp_t;

	mpz_inits(gmp_ct, gmp_ct2, gmp_ct4, gmp_ct6, gmp_ce, 
		  gmp_a, gmp_d, gmp_n, gmp_x, gmp_y, gmp_z, NULL);

	mp_t x, y, z, a, d;

	mp_to_mpz(gmp_n, info->n);

	pthread_mutex_lock(&mutex_gmp_rand);
	mpz_urandomb(gmp_ct, gmprand, BITWIDTH);
	pthread_mutex_unlock(&mutex_gmp_rand);

	mpz_mod(gmp_ct, gmp_ct, gmp_n);

	mpz_mul(gmp_ct2, gmp_ct, gmp_ct);
	mpz_mod(gmp_ct2, gmp_ct2, gmp_n);

	mpz_mul(gmp_ct4, gmp_ct2, gmp_ct2);
	mpz_mod(gmp_ct4, gmp_ct4, gmp_n);

	mpz_mul(gmp_ct6, gmp_ct2, gmp_ct4);
	mpz_mod(gmp_ct6, gmp_ct6, gmp_n);

	mpz_t gmp_tmp;
	mpz_init(gmp_tmp);

	/* e = 3(t^2 -1)*(8t)^-1 */
	mpz_mul_ui(gmp_ce, gmp_ct, 8);
	mpz_invert(gmp_ce, gmp_ce, gmp_n);
	mpz_sub_ui(gmp_tmp, gmp_ct2, 1);
	mpz_mul_ui(gmp_tmp, gmp_tmp, 3);
	mpz_mul(gmp_ce, gmp_tmp, gmp_ce);
	mpz_mod(gmp_ce, gmp_ce, gmp_n);

	/* d = -e^4 */
	mpz_mul(gmp_d, gmp_ce, gmp_ce);
	mpz_mul(gmp_d, gmp_d, gmp_d);
	mpz_sub(gmp_d, gmp_n, gmp_d); // -d = n-d (mod n)
	mpz_mod(gmp_d, gmp_d, gmp_n);

	/* x = 128t^3 * (27t^6 + 63t^4 - 63t^2 - 27)^-1 */
	mpz_mul_ui(gmp_x, gmp_ct6, 27);
	mpz_mul_ui(gmp_tmp, gmp_ct4, 63);
	mpz_add(gmp_x, gmp_x, gmp_tmp);
	mpz_mul_ui(gmp_tmp, gmp_ct2, 63);
	mpz_sub(gmp_x, gmp_x, gmp_tmp);
	mpz_sub_ui(gmp_x, gmp_x, 27);
	mpz_invert(gmp_x, gmp_x, gmp_n);
	mpz_mul(gmp_tmp, gmp_ct, gmp_ct2);
	mpz_mul_ui(gmp_tmp, gmp_tmp, 128);
	mpz_mod(gmp_tmp, gmp_tmp, gmp_n);
	mpz_mul(gmp_x, gmp_x, gmp_tmp);
	mpz_mod(gmp_x, gmp_x, gmp_n);

	/* y = (9t^4 - 2t^2 + 9) * (9t^4 - 9)^-1 */
	mpz_mul_ui(gmp_y, gmp_ct4, 9);
	mpz_add_ui(gmp_y, gmp_y, 9);
	mpz_mul_ui(gmp_tmp, gmp_ct2, 2);
	mpz_sub(gmp_y, gmp_y, gmp_tmp);

	mpz_mul_ui(gmp_tmp, gmp_ct4, 9);
	mpz_sub_ui(gmp_tmp, gmp_tmp, 9);
	mpz_invert(gmp_tmp, gmp_tmp, gmp_n);
	mpz_mul(gmp_y, gmp_tmp, gmp_y);
	mpz_mod(gmp_y, gmp_y, gmp_n);

	/* Transform curve parameters to Montgomery Form */
	mpz_set_si(gmp_a, -1);
	mpz_mod(gmp_a, gmp_a, gmp_n);
	mpz_to_mp(a, gmp_a);
	to_mon(a, a, info);
	mpz_to_mp(d, gmp_d);
	to_mon(d, d, info);

	/* Initialize curve with parameters */
	tw_ed_init_curve(curve, a, d, info);

	// x := y*z = y
	mpz_to_mp(x, gmp_y);
	to_mon(x, x, curve->info);
	mp_copy(p1->x, x);

	// y := x*z = x
	mpz_to_mp(y, gmp_x);
	to_mon(y, y, curve->info);
	mp_copy(p1->y, y);

	// z := x*y
	mpz_mul(gmp_z, gmp_x, gmp_y);
	mpz_mod(gmp_z, gmp_z, gmp_n);
	mpz_to_mp(z, gmp_z);
	to_mon(z, z, curve->info);
	mp_copy(p1->z, z);

	mpz_clear(gmp_ct);
	mpz_clear(gmp_ct2);
	mpz_clear(gmp_ct4);
	mpz_clear(gmp_ct6);
	mpz_clear(gmp_ce);

	mpz_clear(gmp_a);
	mpz_clear(gmp_d);
	mpz_clear(gmp_n);
	mpz_clear(gmp_x);
	mpz_clear(gmp_y);
	mpz_clear(gmp_z);
	mpz_clear(gmp_tmp);

}

void tw_ed_random_curve_gkl2016_j4(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand) {
	/* Curve construction parameters */
	mpz_t gmp_ct, gmp_ct2, gmp_ct3, gmp_ct4, gmp_ct5, gmp_ct6, gmp_ce;
	mpz_inits(gmp_ct, gmp_ct2, gmp_ct3, gmp_ct4, gmp_ct5, gmp_ct6, gmp_ce,
		  gmp_a, gmp_d, gmp_n, gmp_x, gmp_y, gmp_z, NULL);

	/* Curve parameters */
	mpz_t gmp_a, gmp_d, gmp_n;
	/* Point (extended projective coordinates) */
	mpz_t gmp_x, gmp_y, gmp_z;


	mp_t x, y, z, a, d;//, t;

	mp_to_mpz(gmp_n, info->n);

	pthread_mutex_lock(&mutex_gmp_rand);
	mpz_urandomb(gmp_ct, gmprand, BITWIDTH);
	pthread_mutex_unlock(&mutex_gmp_rand);

	mpz_mul(gmp_ct2, gmp_ct, gmp_ct);
	mpz_mod(gmp_ct2, gmp_ct2, gmp_n);
	mpz_mul(gmp_ct3, gmp_ct, gmp_ct2);
	mpz_mod(gmp_ct3, gmp_ct3, gmp_n);
	mpz_mul(gmp_ct4, gmp_ct2, gmp_ct2);
	mpz_mod(gmp_ct4, gmp_ct4, gmp_n);
	mpz_mul(gmp_ct5, gmp_ct2, gmp_ct3);
	mpz_mod(gmp_ct5, gmp_ct5, gmp_n);
	mpz_mul(gmp_ct6, gmp_ct2, gmp_ct4);
	mpz_mod(gmp_ct6, gmp_ct6, gmp_n);

	mpz_t gmp_tmp, gmp_tmp2;
	mpz_inits(gmp_tmp, gmp_tmp2, NULL);

	/* e = (t^2 + 4t)*(t^2 -4)^-1 */
	mpz_mul_ui(gmp_ce, gmp_ct, 4);
	mpz_add(gmp_ce, gmp_ce, gmp_ct2);
	mpz_sub_ui(gmp_tmp, gmp_ct2, 4);
	mpz_invert(gmp_tmp, gmp_tmp, gmp_n);
	mpz_mul(gmp_ce, gmp_tmp, gmp_ce);
	mpz_mod(gmp_ce, gmp_ce, gmp_n);

	/* d = e^4 */
	mpz_mul(gmp_d, gmp_ce, gmp_ce);
	mpz_mod(gmp_d, gmp_d, gmp_n);
	mpz_mul(gmp_d, gmp_d, gmp_d);
	mpz_mod(gmp_d, gmp_d, gmp_n);
	mpz_sub(gmp_d, gmp_n, gmp_d);
	mpz_mod(gmp_d, gmp_d, gmp_n);

	/* x = (2t^3 + 2t^2 - 8t - 8) * (t^4 + 6t^3 + 12t^2 + 16t)^-1 */
	mpz_mul_ui(gmp_x, gmp_ct3, 2);
	mpz_mul_ui(gmp_tmp, gmp_ct2, 2);
	mpz_add(gmp_x, gmp_x, gmp_tmp);
	mpz_mul_ui(gmp_tmp, gmp_ct, 8);
	mpz_sub(gmp_x, gmp_x, gmp_tmp);
	mpz_sub_ui(gmp_x, gmp_x, 8);

	mpz_set(gmp_tmp, gmp_ct4);
	mpz_mul_ui(gmp_tmp2, gmp_ct3, 6);
	mpz_add(gmp_tmp, gmp_tmp, gmp_tmp2);
	mpz_mul_ui(gmp_tmp2, gmp_ct2, 12);
	mpz_add(gmp_tmp, gmp_tmp, gmp_tmp2);
	mpz_mul_ui(gmp_tmp2, gmp_ct, 16);
	mpz_add(gmp_tmp, gmp_tmp, gmp_tmp2);
	mpz_invert(gmp_tmp, gmp_tmp, gmp_n);

	mpz_mul(gmp_x, gmp_x, gmp_tmp);
	mpz_mod(gmp_x, gmp_x, gmp_n);

	/* y = (t^6 + 6t^5 + 10t^4 - 16t^3 - 48t^2 - 32t -32) * (t^6 + 6t^5 + 10t^4 + 16t^3 + 48t^2 + 64t)*/
	mpz_set(gmp_y, gmp_ct6);
	mpz_mul_ui(gmp_tmp, gmp_ct5, 6);
	mpz_add(gmp_y, gmp_y, gmp_tmp);
	mpz_mul_ui(gmp_tmp, gmp_ct4, 10);
	mpz_add(gmp_y, gmp_y, gmp_tmp);
	mpz_mul_ui(gmp_tmp, gmp_ct3, 16);
	mpz_sub(gmp_y, gmp_y, gmp_tmp);
	mpz_mul_ui(gmp_tmp, gmp_ct2, 48);
	mpz_sub(gmp_y, gmp_y, gmp_tmp);
	mpz_mul_ui(gmp_tmp, gmp_ct, 32);
	mpz_sub(gmp_y, gmp_y, gmp_tmp);
	mpz_sub_ui(gmp_y, gmp_y, 32);

	mpz_set(gmp_tmp, gmp_ct6);
	mpz_mul_ui(gmp_tmp2, gmp_ct5, 6);
	mpz_add(gmp_tmp, gmp_tmp, gmp_tmp2);
	mpz_mul_ui(gmp_tmp2, gmp_ct4, 10);
	mpz_add(gmp_tmp, gmp_tmp, gmp_tmp2);
	mpz_mul_ui(gmp_tmp2, gmp_ct3, 16);
	mpz_add(gmp_tmp, gmp_tmp, gmp_tmp2);
	mpz_mul_ui(gmp_tmp2, gmp_ct2, 48);
	mpz_add(gmp_tmp, gmp_tmp, gmp_tmp2);
	mpz_mul_ui(gmp_tmp2, gmp_ct, 64);
	mpz_add(gmp_tmp, gmp_tmp, gmp_tmp2);
	mpz_invert(gmp_tmp, gmp_tmp, gmp_n);

	mpz_mul(gmp_y, gmp_y, gmp_tmp);
	mpz_mod(gmp_y, gmp_y, gmp_n);

	/* Transform curve parameters to Montgomery Form */
	mpz_set_si(gmp_a, -1);
	mpz_mod(gmp_a, gmp_a, gmp_n);
	mpz_to_mp(a, gmp_a);
	to_mon(a, a, info);
	mpz_to_mp(d, gmp_d);
	to_mon(d, d, info);

	/* Initialize curve with parameters */
	tw_ed_init_curve(curve, a, d, info);

	// x := y*z = y
	mpz_to_mp(x, gmp_y);
	to_mon(x, x, curve->info);
	mp_copy(p1->x, x);

	// y := x*z = x
	mpz_to_mp(y, gmp_x);
	to_mon(y, y, curve->info);
	mp_copy(p1->y, y);

	// z := x*y
	mpz_mul(gmp_z, gmp_x, gmp_y);
	mpz_mod(gmp_z, gmp_z, gmp_n);
	mpz_to_mp(z, gmp_z);
	to_mon(z, z, curve->info);
	mp_copy(p1->z, z);

	mpz_clear(gmp_ct);
	mpz_clear(gmp_ct2);
	mpz_clear(gmp_ct3);
	mpz_clear(gmp_ct4);
	mpz_clear(gmp_ct5);
	mpz_clear(gmp_ct6);
	mpz_clear(gmp_ce);

	mpz_clear(gmp_a);
	mpz_clear(gmp_d);
	mpz_clear(gmp_n);
	mpz_clear(gmp_x);
	mpz_clear(gmp_y);
	mpz_clear(gmp_z);
	mpz_clear(gmp_tmp);

}

void tw_ed_point_set_id(point_tw_ed *p) {
	mp_set_ui(p->x, 0);
	mp_set_ui(p->y, 1);
	mp_set_ui(p->z, 1);
}

__inline__
void tw_ed_point_invert_precomp(point_tw_ed *p, const mon_info *info) {
	tw_ed_point_invert(p->x, info->n, p->x);
}

#ifdef __cplusplus
}
#endif

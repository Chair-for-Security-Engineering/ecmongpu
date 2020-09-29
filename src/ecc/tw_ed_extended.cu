#include <cuda.h>
#include <cuda_runtime.h>
#include "ecc/twisted_edwards.h"
//#include "ecm/factor_task.h"
#include "mp/mp.h"
#include "log.h"
#include "mp/gmp_conversion.h"
#include "ecc/twisted_edwards.h"
#include "mp/mp_montgomery.h"


#ifdef __cplusplus
extern "C" {
#endif

/**
 * See https://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#doubling-dbl-2008-hwcd
 *
 * A = X^2
 * B = Y^2
 * C = 2*Z^2
 * D = a*A
 * E = (X+Y)2-A-B
 * G = D+B
 * F = G-C
 * H = D-B
 * X3 = E*F
 * Y3 = G*H
 * T3 = E*H
 * Z3 = F*G
 */
__host__ __device__
int tw_ed_double(point_tw_ed *r, const point_tw_ed *op, const curve_tw_ed *curve, const mon_info *info, bool extend) {

	mp_t a, b, c, d, e, f, g, h;

	//  A = X^2
	mon_square(a, op->x, info);

	//  B = Y^2
	mon_square(b, op->y, info);

	//  C = 2*Z^2 = Z^2 + Z^2
	mon_square(c, op->z, info);
	mp_add_mod(c, c, c, info->n);

	//  D = a*A = -1*A = N-A
	mp_sub(d, info->n, a);


	//  E = (X+Y)^2-A-B
	mp_add_mod(e, op->x, op->y, info->n);
	mon_square(e, e, info);
	mp_sub_mod(e, e, a, info->n);
	mp_sub_mod(e, e, b, info->n);

	//  G = D+B
	mp_add_mod(g, d, b, info->n);

	//  F = G-C
	mp_sub_mod(f, g, c, info->n);

	//  H = D-B
	mp_sub_mod(h, d, b, info->n);

	//  X3 = E*F
	mon_prod_distinct(r->x, e, f, info);
	//  Y3 = G*H
	mon_prod_distinct(r->y, g, h, info);
	//  Z3 = F*G
	mon_prod_distinct(r->z, f, g, info);

	if (extend) {
		//  T3 = E*H
		mon_prod_distinct(r->t, e, h, info);
	}


	return 0;
}

/*
  See: https://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#tripling-tpl-2015-c

     YY = Y1^2
     aXX = a*X1^2
     Ap = YY+aXX
     B = 2*(2*Z1^2-Ap)
     xB = aXX*B
     yB = YY*B
     AA = Ap*(YY-aXX)
     F = AA-yB
     G = AA+xB
     xE = X1*(yB+AA)
     yH = Y1*(xB-AA)
     zF = Z1*F
     zG = Z1*G
     X3 = xE*zF
     Y3 = yH*zG
     Z3 = zF*zG
     T3 = xE*yH
*/
__host__ __device__
int tw_ed_triple(point_tw_ed *r, const point_tw_ed *op, const curve_tw_ed *curve, const mon_info *info, bool extend) {

	mp_t yy, axx, ap, b, xb, yb, aa, f, g, xe, yh, zf, zg;

     //YY = Y1^2
	mon_square(yy, op->y, info);

     //aXX = a*X1^2
	mon_square(axx, op->x, info);
	mp_sub(axx, info->n, axx);

     //Ap = YY+aXX
	mp_add_mod(ap, yy, axx, info->n);

     //B = 2*(2*Z1^2-Ap)
  mon_square(b, op->z, info);
  mp_add_mod(b, b, b, info->n);
  mp_sub_mod(b, b, ap, info->n);
  mp_add_mod(b, b, b, info->n);

     //xB = aXX*B
  mon_prod_distinct(xb, axx, b, info);

     //yB = YY*B
  mon_prod_distinct(yb, yy, b, info);

     //AA = Ap*(YY-aXX)
  mp_sub_mod(aa, yy, axx, info->n);
  mon_prod(aa, aa, ap, info);

     //F = AA-yB
  mp_sub_mod(f, aa, yb, info->n);
     //G = AA+xB
  mp_add_mod(g, aa, xb, info->n);
     //xE = X1*(yB+AA)
  mp_add_mod(xe, yb, aa, info->n);
  mon_prod(xe, xe, op->x, info);
     //yH = Y1*(xB-AA)
  mp_sub_mod(yh, xb, aa, info->n);
  mon_prod(yh, yh, op->y, info);

  if(!extend){
     //X3 = xE*F
  mon_prod_distinct(r->x, xe, f, info);
     //Y3 = yH*G
  mon_prod_distinct(r->y, yh, g, info);
     //Z3 = Z*F*G
  mon_prod_distinct(zg, op->z, g, info);
  mon_prod_distinct(r->z, zg, f, info);
  }

  if (extend) {
     //zF = Z1*F
  mon_prod_distinct(zf, f, op->z, info);

     //zG = Z1*G
  mon_prod_distinct(zg, g, op->z, info);

     //X3 = xE*zF
  mon_prod_distinct(r->x, xe, zf, info);
     //Y3 = yH*zG
  mon_prod_distinct(r->y, yh, zg, info);
     //Z3 = zF*zG
  mon_prod_distinct(r->z, zf, zg, info);

    //T3 = xE*yH
		mon_prod_distinct(r->t, xe, yh, info);
	}
	return 0;
}



void tw_ed_point_invert(point_tw_ed *p, const mon_info *info) {
	mp_sub(p->x, info->n, p->x);
	mp_sub(p->t, info->n, p->t);
}




/** See https://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html#addition-add-2008-hwcd-3
 *
 *  Strongly unified: Also correct for P1 == P2: P3 = P1 + P2 = P1*2
 *
 *    A = (Y1-X1)*(Y2-X2)
 *    B = (Y1+X1)*(Y2+X2)
 *    C = T1*k*T2
 *    E = B-A
 *    F = D-C
 *    G = D+C
 *    H = B+A
 *    X3 = E*F
 *    Y3 = G*H
 *    T3 = E*H
 *    Z3 = F*G
 */
__host__ __device__
int tw_ed_add(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
			  const mon_info *info, bool extend) {

	mp_t a, b, c, d, e, f, g, h;
	mp_t tmp;

	//    A = (Y1-X1)*(Y2-X2)
	mp_sub_mod(tmp, op1->y, op1->x, info->n);
	mp_sub_mod(a, op2->y, op2->x, info->n);
	mon_prod(a, a, tmp, info);

	//    B = (Y1+X1)*(Y2+X2)
	mp_add_mod(tmp, op1->y, op1->x, info->n);
	mp_add_mod(b, op2->y, op2->x, info->n);
	mon_prod(b, b, tmp, info);

	//    C = T1*k*T2
	mon_prod(c, op1->t, op2->t, info);
	mon_prod(c, c, curve->k, info);

	//    D = Z1*2*Z2 = (Z1+Z2) + (Z1+Z2)
	mon_prod_distinct(d, op1->z, op2->z, info);
	mp_add_mod(d, d, d, info->n);

	//    E = B-A
	mp_sub_mod(e, b, a, info->n);

	//    F = D-C
	mp_sub_mod(f, d, c, info->n);

	//    G = D+C
	mp_add_mod(g, d, c, info->n);

	//    H = B+A
	mp_add_mod(h, b, a, info->n);

	//    X3 = E*F
	mon_prod_distinct(r->x, e, f, info);

	//    Y3 = G*H
	mon_prod_distinct(r->y, g, h, info);

	//    Z3 = F*G
	mon_prod_distinct(r->z, f, g, info);

	if (extend) {
		//    T3 = E*H
		mon_prod_distinct(r->t, e, h, info);
	}

	return 0;
}


#ifdef OPTIMIZE_PRECOMP
/* precomputed point has 
  Xp := (Y2-X2)
  Yp := (Y2+X2)
  Z_p := 1
  T_p := k*T
	mp_sub_mod(a, op1->y, op1->x, info->n);
	mp_add_mod(b, op2->y, op2->x, info->n);
*/
__inline__
__host__ __device__
int tw_ed_add_precomp(point_tw_ed *r, const point_tw_ed *op1, const point_tw_ed *op2, const curve_tw_ed *curve,
					  const mon_info *info, bool extend) {

	mp_t a, b, c, d, e, f, g, h;

	//    A = (Y1-X1)*Xp
	mp_sub_mod(a, op1->y, op1->x, info->n);
	mon_prod(a, a, op2->x, info);

	//    B = (Y1+X1)*Yp
	mp_add_mod(b, op1->y, op1->x, info->n);
	mon_prod(b, b, op2->y, info);

	//    C = T1*Tp
	mon_prod_distinct(c, op1->t, op2->t, info);

	//    D = Z1*2*Zp = Z1 + Z1
	mp_add_mod(d, op1->z, op1->z, info->n);

	//    E = B-A
	mp_sub_mod(e, b, a, info->n);

	//    F = D-C
	mp_sub_mod(f, d, c, info->n);

	//    G = D+C
	mp_add_mod(g, d, c, info->n);

	//    H = B+A
	mp_add_mod(h, b, a, info->n);

	//    X3 = E*F
	mon_prod_distinct(r->x, e, f, info);

	//    Y3 = G*H
	mon_prod_distinct(r->y, g, h, info);

	//    Z3 = F*G
	mon_prod_distinct(r->z, f, g, info);

	if (extend) {
		//    T3 = E*H
		mon_prod_distinct(r->t, e, h, info);
	}

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
	mp_printf("\td: %Zi\n", c->d);
	mp_printf("\tk: %Zi\n", c->k);

	mp_t tmp;

	printf("  Normal:\n");
	from_mon(tmp, c->d, c->info);
	mp_printf("\td: %Zi\n", tmp);
	from_mon(tmp, c->k, c->info);
	mp_printf("\tk: %Zi\n", tmp);
}


__host__ __device__
int tw_ed_copy_point(point_tw_ed *dest, const point_tw_ed *src) {
	mp_copy(dest->x, src->x);
	mp_copy(dest->y, src->y);
	mp_copy(dest->z, src->z);
	mp_copy(dest->t, src->t);
	return 0;
}

__host__ __device__
int tw_ed_copy_point_sc(point_tw_ed_strided *dest, const size_t dest_elem,
						const point_tw_ed *src) {
	mp_copy_sc(dest->x, dest_elem, src->x);
	mp_copy_sc(dest->y, dest_elem, src->y);
	mp_copy_sc(dest->z, dest_elem, src->z);
	mp_copy_sc(dest->t, dest_elem, src->t);
	return 0;
}

__host__ __device__
int tw_ed_copy_point_cs(point_tw_ed *dest,
						const point_tw_ed_strided *src, const size_t src_elem) {
	mp_copy_cs(dest->x, src->x, src_elem);
	mp_copy_cs(dest->y, src->y, src_elem);
	mp_copy_cs(dest->z, src->z, src_elem);
	mp_copy_cs(dest->t, src->t, src_elem);
	return 0;
}

__host__ __device__
int tw_ed_copy_point_ss(point_tw_ed_strided *dest, const size_t dest_elem,
						const point_tw_ed_strided *src, const size_t src_elem) {
	mp_copy_ss(dest->x, dest_elem, src->x, src_elem);
	mp_copy_ss(dest->y, dest_elem, src->y, src_elem);
	mp_copy_ss(dest->z, dest_elem, src->z, src_elem);
	mp_copy_ss(dest->t, dest_elem, src->t, src_elem);
	return 0;
}


void tw_ed_print_point(const point_tw_ed *p, const mon_info *info) {
	printf("Point:\n");
	printf("  Montgomery:\n");
	mp_printf("\tx: %Zi\n", p->x);
	mp_printf("\ty: %Zi\n", p->y);
	mp_printf("\tz: %Zi\n", p->z);
	mp_printf("\tt: %Zi\n", p->t);

	mp_t tmp;
	printf("  Normal:\n");
	from_mon(tmp, p->x, info);
	mp_printf("\tx: %Zi\n", tmp);
	from_mon(tmp, p->y, info);
	mp_printf("\ty: %Zi\n", tmp);
	from_mon(tmp, p->z, info);
	mp_printf("\tz: %Zi\n", tmp);
	from_mon(tmp, p->t, info);
	mp_printf("\tt: %Zi\n", tmp);
}


__host__ __device__
void tw_ed_to_reg(point_tw_ed *p, const mon_info *info) {
	(void) p;
	(void) info;
}


bool tw_ed_scale_point(point_tw_ed *r, const point_tw_ed *p, const mon_info *info) {
	mp_t zinv;
	mon_inv(zinv, p->z, info);

	mon_prod(r->x, p->x, zinv, info);
	mon_prod(r->y, p->y, zinv, info);
	mp_set_ui(r->z, 1);
	mon_prod(r->t, r->x, r->y, info);

	return true;
}

void tw_ed_random_curve_naive(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand, void *data) {

	/* Curve parameters */
	mpz_t gmp_a, gmp_d, gmp_n;
	/* Point (extended projective coordinates) */
	mpz_t gmp_x, gmp_y, gmp_z, gmp_t;

	mpz_inits(gmp_a, gmp_d, gmp_n, gmp_x, gmp_y, gmp_z, gmp_t, NULL);

	mp_t x, y, a, d, t;

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

	/* Calculate t = xy (extended coordinates) */
	mpz_mul(gmp_t, gmp_x, gmp_y);
	mpz_mod(gmp_t, gmp_t, gmp_n);

	/* Transform curve parameters to Montgomery Form */
	mpz_set_si(gmp_a, -1);
	mpz_mod(gmp_a, gmp_a, gmp_n);
	mpz_to_mp(a, gmp_a);
	to_mon(a, a, info);
	mpz_to_mp(d, gmp_d);
	to_mon(d, d, info);

	/* Initialize curve with parameters */
	tw_ed_init_curve(curve, a, d, info);

	mpz_to_mp(t, gmp_t);
	to_mon(t, t, curve->info);
	mp_copy(p1->t, t);

	mpz_to_mp(x, gmp_x);
	to_mon(x, x, curve->info);
	mp_copy(p1->x, x);

	mpz_to_mp(y, gmp_y);
	to_mon(y, y, curve->info);
	mp_copy(p1->y, y);

	mp_set_ui(p1->z, 1);
	to_mon(p1->z, p1->z, curve->info);

	mpz_clears(gmp_a, gmp_d, gmp_n, gmp_x, gmp_y, gmp_z, gmp_t, gmp_tmp, gmp_x2, gmp_y2, NULL);
}


void tw_ed_random_curve_gkl2016_j1(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand, void *data) {

	/* Curve construction parameters */
	mpz_t gmp_ct, gmp_ct2, gmp_ct4, gmp_ct6, gmp_ce;
	mpz_inits(gmp_ct, gmp_ct2, gmp_ct4, gmp_ct6, gmp_ce, NULL);

	/* Curve parameters */
	mpz_t gmp_a, gmp_d, gmp_n;
	/* Point (extended projective coordinates) */
	mpz_t gmp_x, gmp_y, gmp_z, gmp_t;

	mpz_inits(gmp_a, gmp_d, gmp_n, gmp_x, gmp_y, gmp_z, gmp_t, NULL);

	mp_t x, y, a, d, t;

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

	/* Calculate t = xy (extended coordinates) */
	mpz_mul(gmp_t, gmp_x, gmp_y);
	mpz_mod(gmp_t, gmp_t, gmp_n);

	/* Transform curve parameters to Montgomery Form */
	mpz_set_si(gmp_a, -1);
	mpz_mod(gmp_a, gmp_a, gmp_n);
	mpz_to_mp(a, gmp_a);
	to_mon(a, a, info);
	mpz_to_mp(d, gmp_d);
	to_mon(d, d, info);

	/* Initialize curve with parameters */
	tw_ed_init_curve(curve, a, d, info);

	mpz_to_mp(t, gmp_t);
	to_mon(t, t, curve->info);
	mp_copy(p1->t, t);

	mpz_to_mp(x, gmp_x);
	to_mon(x, x, curve->info);
	mp_copy(p1->x, x);

	mpz_to_mp(y, gmp_y);
	to_mon(y, y, curve->info);
	mp_copy(p1->y, y);

	mp_set_ui(p1->z, 1);
	to_mon(p1->z, p1->z, curve->info);

	mpz_clears(gmp_ct, gmp_ct2, gmp_ct4, gmp_ct6, gmp_ce,
		   gmp_a, gmp_d, gmp_n, gmp_x, gmp_y, gmp_z, gmp_t, gmp_tmp, NULL);

}

void gkl2016_dbl(point_gkl2016 *p, mpz_t mod){
	mpz_t lambda, tmp, tmpx, tmpy;
	mpz_inits(lambda, tmpx, tmpy, tmp, NULL);

	// lambda = (3* (x**2) - 2*x - 9) * (2*y)**(-1)
	mpz_mul(lambda, p->x, p->x);
	mpz_mul_ui(lambda, lambda, 3);
	mpz_sub(lambda, lambda, p->x);
	mpz_sub(lambda, lambda, p->x);
	mpz_sub_ui(lambda, lambda, 9);
	mpz_mod(lambda, lambda, mod);
	mpz_add(tmp, p->y, p->y);
	mpz_invert(tmp, tmp, mod);
	mpz_mul(lambda, lambda, tmp);
	mpz_mod(lambda, lambda, mod);

	//x3 = lambda**2 + 1 - 2*x
	mpz_mul(tmpx, lambda, lambda);
	mpz_add_ui(tmpx, tmpx, 1);
	mpz_sub(tmpx, tmpx, p->x);
	mpz_sub(tmpx, tmpx, p->x);
	mpz_mod(tmpx, tmpx, mod);

	//y3 = -lambda*x3 + l*x - y
	mpz_mul(tmpy, lambda, tmpx);
	mpz_sub(tmpy, mod, tmpy);
	mpz_mul(tmp, lambda, p->x);
	mpz_add(tmpy, tmpy, tmp);
	mpz_sub(tmpy, tmpy, p->y);
	mpz_mod(tmpy, tmpy, mod);

	mpz_set(p->x, tmpx);
	mpz_set(p->y, tmpy);

	mpz_clears(lambda, tmp, tmpx, tmpy, NULL);
}

void gkl2016_addbase(point_gkl2016 *p, mpz_t mod){
	mpz_t lambda, tmp, tmpx, tmpy;
	mpz_inits(lambda, tmpx, tmpy, tmp, NULL);

	// lambda = (8-y) * (5-x)**(-1)
	mpz_sub(lambda, mod, p->y);
	mpz_add_ui(lambda, lambda, 8);
	mpz_sub(tmp, mod, p->x);
	mpz_add_ui(tmp, tmp, 5);
	mpz_invert(tmp, tmp, mod);
	mpz_mul(lambda, lambda, tmp);
	mpz_mod(lambda, lambda, mod);

	// x3 = (lambda ** 2) - 4 - x
	mpz_mul(tmpx, lambda, lambda);
	mpz_sub_ui(tmpx, tmpx, 4);
	mpz_sub(tmpx, tmpx, p->x);
	mpz_mod(tmpx, tmpx, mod);

	//y3 = -lambda*x3 + l*x - y
	mpz_mul(tmpy, lambda, tmpx);
	mpz_sub(tmpy, mod, tmpy);
	mpz_mul(tmp, lambda, p->x);
	mpz_mod(tmp, tmp, mod);
	mpz_add(tmpy, tmpy, tmp);
	mpz_sub(tmpy, tmpy, p->y);
	mpz_mod(tmpy, tmpy, mod);

	mpz_set(p->x, tmpx);
	mpz_set(p->y, tmpy);

	mpz_clears(lambda, tmp, tmpx, tmpy, NULL);
}

void gkl2016_basesmul(point_gkl2016 *p, uint64_t rand, mpz_t mod){
	mpz_set_ui(p->x, 5);
	mpz_set_ui(p->y, 8);

	uint64_t bit = 0x100000000000000; //highest bit set

	// shift over leading zeros
	while(! (rand & bit)) bit >>= 1;

	// shift over leading one
	bit >>= 1;

	// double and add
	while(bit){
		gkl2016_dbl(p, mod);
		if (rand & bit){
			gkl2016_addbase(p, mod);
		}
		bit >>= 1;
	}
	LOG_VERBOSE("gkl2016 base * %i.\tp.x: %Zi\tp.y: %Zi\n", rand, p->x, p->y);

}

void gkl2016_get_t(mpz_t ct, point_gkl2016 *p,  mpz_t gmp_n){
	// t = (4x + 4) * (y-4)**(-1)
	mpz_t tmp;
	mpz_init(tmp);

	mpz_mul_ui(ct, p->x, 4);
	mpz_add_ui(ct, ct, 4);
	mpz_sub_ui(tmp, p->y, 4);
	mpz_invert(tmp, tmp, gmp_n);
	mpz_mul(ct, ct, tmp);
	mpz_mod(ct, ct, gmp_n);

	mpz_clear(tmp);
}



void tw_ed_random_curve_gkl2016_j4(point_tw_ed *p1, curve_tw_ed *curve, mon_info *info, gmp_randstate_t gmprand, void* data) {

	point_gkl2016 *phelper = (point_gkl2016*)data;

	/* Curve construction parameters */
	mpz_t gmp_ct, gmp_ct2, gmp_ct3, gmp_ct4, gmp_ct5, gmp_ct6, gmp_ce;

	/* Curve parameters */
	mpz_t gmp_a, gmp_d, gmp_n;
	/* Point (extended projective coordinates) */
	mpz_t gmp_x, gmp_y, gmp_z, gmp_t;

	mpz_inits(gmp_ct, gmp_ct2, gmp_ct3, gmp_ct4, gmp_ct5, gmp_ct6, gmp_ce,
		  gmp_a, gmp_d, gmp_n, gmp_x, gmp_y, gmp_z, gmp_t, NULL);

	mp_t x, y, a, d, t;

	mp_to_mpz(gmp_n, info->n);

  	pthread_mutex_lock(&phelper->mutex);
	gkl2016_addbase(phelper, gmp_n);
  	pthread_mutex_unlock(&phelper->mutex);
	gkl2016_get_t(gmp_ct, phelper, gmp_n);
  
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

	/* Calculate t = xy (extended coordinates) */
	mpz_mul(gmp_t, gmp_x, gmp_y);
	mpz_mod(gmp_t, gmp_t, gmp_n);

	/* Transform curve parameters to Montgomery Form */
	mpz_set_si(gmp_a, -1);
	mpz_mod(gmp_a, gmp_a, gmp_n);
	mpz_to_mp(a, gmp_a);
	to_mon(a, a, info);
	mpz_to_mp(d, gmp_d);
	to_mon(d, d, info);

	/* Initialize curve with parameters */
	tw_ed_init_curve(curve, a, d, info);

	mpz_to_mp(t, gmp_t);
	to_mon(t, t, curve->info);
	mp_copy(p1->t, t);

	mpz_to_mp(x, gmp_x);
	to_mon(x, x, curve->info);
	mp_copy(p1->x, x);

	mpz_to_mp(y, gmp_y);
	to_mon(y, y, curve->info);
	mp_copy(p1->y, y);

	mp_set_ui(p1->z, 1);
	to_mon(p1->z, p1->z, curve->info);

	mpz_clears(gmp_ct, gmp_ct2, gmp_ct3, gmp_ct4, gmp_ct5, gmp_ct6, gmp_ce, 
		   gmp_a, gmp_d, gmp_n, gmp_x, gmp_y, gmp_z, gmp_t, gmp_tmp, NULL);

}

void tw_ed_point_set_id(point_tw_ed *p) {
	mp_set_ui(p->x, 0);
	mp_set_ui(p->y, 1);
	mp_set_ui(p->z, 1);
	mp_set_ui(p->t, 0);
}

#ifdef OPTIMIZE_PRECOMP
void tw_ed_point_invert_precomp(point_tw_ed *p, const mon_info *info) {
	mp_switch(p->y, p->x);
	mp_sub(p->t, info->n, p->t);
}
#else 
__inline__
void tw_ed_point_invert_precomp(point_tw_ed *p, const mon_info *info) {
	tw_ed_point_invert(p, info);
}
#endif /* OPTIMIZE_PRECOMP */

#ifdef __cplusplus
}
#endif

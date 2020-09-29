#include <gmp.h>
#include <time.h>
#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"
#include "log.h"

int test() {
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	srand(time(NULL));
	gmp_randseed_ui(gmprand, rand());

	for (int i = 0; i < TEST_RUNS * 1000; i++) {

		mpz_t gmp_a, gmp_b, gmp_n;
		mpz_t gmp_r, gmp_R, gmp_result_mp, gmp_Rsq, gmp_result;

		mpz_init(gmp_a);
		mpz_init(gmp_b);
		mpz_init(gmp_n);
		mpz_init(gmp_R);
		mpz_init(gmp_r);
		mpz_init(gmp_Rsq);

		mpz_init(gmp_result_mp);
		mpz_init(gmp_result);

		mp_t a, b, n, r;

		mpz_set_ui(gmp_r, 2);
		mpz_pow_ui(gmp_r, gmp_r, LIMB_BITS);

		mpz_pow_ui(gmp_R, gmp_r, LIMBS);
		LOG_VERBOSE("gmp_R: %Zi\n", gmp_R);
		mpz_pow_ui(gmp_Rsq, gmp_r, LIMBS * 2);


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

#ifdef LOG_VERBOSE_ENABLED
		print_info(&info);
#endif

		/* GMP's montgomery transformation */
		mpz_t gmp_a_mon, gmp_b_mon;
		mpz_init(gmp_a_mon);
		mpz_init(gmp_b_mon);
		/* a_mon = R * a mod n */
		mpz_mul(gmp_a_mon, gmp_R, gmp_a);
		mpz_mod(gmp_a_mon, gmp_a_mon, gmp_n);

		/* b_mon = R * b mod n */
		mpz_mul(gmp_b_mon, gmp_R, gmp_b);
		mpz_mod(gmp_b_mon, gmp_b_mon, gmp_n);


		/* Operands */
		mpz_to_mp(a, gmp_a);
		mpz_to_mp(b, gmp_b);
#ifdef LOG_LEVEL_VERBOSE_ENABLED
		print_info(&info);
		gmp_printf("gmp_a : %Zx\n", gmp_a);
		gmp_printf("a_mon (GMP) %Zi\n", gmp_a_mon);
		gmp_printf("gmp_b : %Zx\n", gmp_b);
		gmp_printf("b_mon (GMP) %Zi\n", gmp_b_mon);
		mp_printf("a: %Zi\n", a);
		mp_printf("b: %Zi\n", b);

		/* To montgomery form */
		LOG_VERBOSE("Converting a to Montgomery Form:\n");
#endif

		/* Compute Montgomery Transform of b and b */
		LOG_VERBOSE("Montgomery Transform (a):\n");
		to_mon(a, a, &info);
#ifdef LOG_LEVEL_VERBOSE_ENABLED
		mp_printf("a_mon %Zi\n", a);
#endif
		LOG_VERBOSE("Montgomery Transform (b):\n");
		to_mon(b, b, &info);
#ifdef LOG_LEVEL_VERBOSE_ENABLED
		mp_printf("b_mon %Zi\n", b);
#endif

		/* Compute GMP's Montgomery Transform of a and check */
		mpz_t gmp_mp_mon_a, tmp;
		mpz_init(gmp_mp_mon_a);
		mpz_init(tmp);
		mp_to_mpz(gmp_mp_mon_a, a);
		mpz_sub(tmp, gmp_a_mon, gmp_mp_mon_a);
		if (mpz_cmp_ui(tmp, 0) != 0) {
			printf("MP Library size:  %ld\n", mpz_sizeinbase(gmp_mp_mon_a, 2));
			printf("GMP Library size: %ld\n", mpz_sizeinbase(gmp_a_mon, 2));
			gmp_printf("MonProd Conversion for a failed. Difference: %Zx\n", tmp);
			gmp_printf("MonProd Conversion for a failed. Difference: %Zi\n", tmp);
			TEST_FAILURE;
		}
		//mpz_clear(gmp_a_mon);
		//mpz_clear(gmp_mp_mon_a);


		/* Compute GMP's Montgomery Transform of b and check */
		mpz_t gmp_mp_mon_b;
		mpz_init(gmp_mp_mon_b);
		mp_to_mpz(gmp_mp_mon_b, b);
		mpz_sub(tmp, gmp_b_mon, gmp_mp_mon_b);
		if (mpz_cmp_ui(tmp, 0) != 0) {
			printf("MonProd Conversion for b failed.\n");
			gmp_printf("GMP Library:  %Zi\n", gmp_b_mon);
			gmp_printf("MP Library:  %Zi\n", gmp_mp_mon_b);
			printf("GMP Library size: %ld\n", mpz_sizeinbase(gmp_b_mon, 2));
			printf("MP Library size:  %ld\n", mpz_sizeinbase(gmp_mp_mon_b, 2));
			printf("GMP Library size: %ld\n", mpz_sizeinbase(gmp_b_mon, 2));
			gmp_printf("Difference: %Zx\n", tmp);
			gmp_printf("Difference: %Zi\n", tmp);
			TEST_FAILURE;
		}
		//mpz_clear(tmp);
		//mpz_clear(gmp_b_mon);
		//mpz_clear(gmp_mp_mon_b);

		/* Compute Montgomery Product */
		LOG_VERBOSE("\n=== Multiply:\n");
		mon_prod(r, a, b, &info);

		/* Compute gmp's result r = a * b mod n */
		mpz_mul(gmp_result, gmp_a, gmp_b);
		mpz_mod(gmp_result, gmp_result, gmp_n);

		/* r_mon = R * r mod n */
		mpz_t r_mon;
		mpz_init(r_mon);
		mpz_mul(r_mon, gmp_R, gmp_result);
		mpz_mod(r_mon, r_mon, gmp_n);
#ifdef LOG_VERBOSE_ENABLED
		gmp_printf("GMP r (mon): %Zi\n", r_mon);
		mp_printf("MP r (mon):  %Zi\n", r);
#endif

		LOG_VERBOSE("\n=== Inverse Transform:\n");
		from_mon(r, r, &info);






		/* Convert result to GMP */
		mp_to_mpz(gmp_result_mp, r);


		if (mpz_cmp(gmp_result, gmp_result_mp) != 0) {
			mpz_t tmp;
			mpz_init(tmp);
			mpz_sub(tmp, gmp_result, gmp_result_mp);
			LOG_FATAL("MonProd Test failed");
		       	LOG_FATAL("\tGMP: %Zi", gmp_result);
		       	LOG_FATAL("\tMP:  %Zi", gmp_result_mp);
		       	LOG_FATAL("\tGMP: %Zx", gmp_result);
		       	LOG_FATAL("\tMP:  %Zx", gmp_result_mp);
			LOG_FATAL("\tDifference: %Zx", tmp);
			LOG_FATAL("\tDifference: %Zi", tmp);
			mpz_clear(tmp);
			TEST_FAILURE;
		}

#ifdef LOG_LEVEL_VERBOSE_ENABLED
		LOG_VERBOSE("MonProd Test Passed.\n");
		mp_printf("\tmp_res:  %Zi\n", r);
		LOG_VERBOSE("\tgmp_res: %Zi\n", gmp_result);
#endif

		/* Cleanup GMP */
		mpz_clear(gmp_a);
		mpz_clear(gmp_b);
		mpz_clear(gmp_n);
		mpz_clear(gmp_R);
		mpz_clear(gmp_r);
		mpz_clear(gmp_Rsq);
		mpz_clear(gmp_result_mp);
		mpz_clear(gmp_result);
	}
	gmp_randclear(gmprand);

	TEST_SUCCESS;
}


TEST_MAIN;


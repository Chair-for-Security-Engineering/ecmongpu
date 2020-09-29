#include <gmp.h>
#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"
#include "log.h"

int test() {
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	for (int i = 0; i < TEST_RUNS; i++) {

		mpz_t gmp_a, gmp_n, gmp_e;
		mpz_t gmp_result_mp, gmp_result;

		mpz_init(gmp_a);
		mpz_init(gmp_n);
		mpz_init(gmp_e);

		mpz_init(gmp_result_mp);
		mpz_init(gmp_result);

		mp_t a, n, e, r;

		mpz_urandomb(gmp_n, gmprand, BITWIDTH);
		/* n has to be an odd modulus */
		mpz_setbit(gmp_n, 0);
		mpz_urandomb(gmp_a, gmprand, BITWIDTH);
		mpz_urandomb(gmp_e, gmprand, BITWIDTH);
		//mpz_set_ui(gmp_e, 0);


		// Reduce operands mod n (to be safe)
		mpz_mod(gmp_a, gmp_a, gmp_n);
		mpz_mod(gmp_e, gmp_e, gmp_n);

		mpz_to_mp(n, gmp_n);

		/* Allocate and calculate Montgomery Constants for this modulus */
		mon_info info;
		mon_info_populate(n, &info);

		/* Operands */
		mpz_to_mp(a, gmp_a);
		mpz_to_mp(e, gmp_e);

		/* Compute Montgomery Transform of b and b */
		LOG_VERBOSE("Montgomery Transform (a):\n");
		to_mon(a, a, &info);

		/* Compute Montgomery Product */
		LOG_VERBOSE("\n=== Multiply:\n");
		mon_exp(r, a, e, &info);

		LOG_VERBOSE("\n=== Inverse Transform:\n");
		from_mon(r, r, &info);

		/* Compute gmp's result r = a * b mod n */
		mpz_powm(gmp_result, gmp_a, gmp_e, gmp_n);

		/* Convert result to GMP */
		mp_to_mpz(gmp_result_mp, r);


		if (mpz_cmp(gmp_result, gmp_result_mp) != 0) {
			mpz_t tmp;
			mpz_init(tmp);
			mpz_sub(tmp, gmp_result, gmp_result_mp);
			LOG_FATAL("MonExp Test failed: %Zi ~= %Zi\n\tDifference: %Zx", gmp_result, gmp_result_mp, tmp);
			mpz_clear(tmp);
			TEST_FAILURE;
		}

#ifdef LOG_LEVEL_VERBOSE_ENABLED
		LOG_VERBOSE("MonExp Test Passed.\n");
		mp_printf("\tmp_res:  %Zi\n", r);
		gmp_printf("\tgmp_res: %Zi\n", gmp_result);
#endif

		/* Cleanup GMP */
		mpz_clear(gmp_a);
		mpz_clear(gmp_e);
		mpz_clear(gmp_n);
		mpz_clear(gmp_result_mp);
		mpz_clear(gmp_result);
	}

	gmp_randclear(gmprand);

	TEST_SUCCESS;
}

TEST_MAIN;


#include <gmp.h>
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"


int test() {
	gmp_randstate_t rand;
	gmp_randinit_default(rand);

	mpz_t gmp_a, gmp_b, gmp_r, gmp_r2, gmp_n;
	mpz_init(gmp_a);
	mpz_init(gmp_b);
	mpz_init(gmp_n);
	mpz_init(gmp_r);
	mpz_init(gmp_r2);

	mp_t a, b, r, n;

	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_urandomb(gmp_n, rand, BITWIDTH);
		mpz_nextprime(gmp_n, gmp_n);

	  mpz_urandomb(gmp_a, rand, BITWIDTH);
		mpz_urandomb(gmp_b, rand, BITWIDTH);

		mpz_mod(gmp_a, gmp_a, gmp_n);
		mpz_mod(gmp_b, gmp_b, gmp_n);

		mpz_to_mp(n, gmp_n);
		mpz_to_mp(a, gmp_a);
		mpz_to_mp(b, gmp_b);

		mp_sub_mod(r, a, b, n);
		mpz_sub(gmp_r, gmp_a, gmp_b);
		mpz_mod(gmp_r, gmp_r, gmp_n);
#ifdef LOG_VERBOSE_ENABLED
		gmp_printf("N:\t%Zi\nA:\t%Zi\nB:\t%Zi\n", gmp_n, gmp_a, gmp_b);
		gmp_printf("A-B:\t%Zi\n", gmp_r);
#endif

		mp_to_mpz(gmp_r2, r);

		if (mpz_cmp(gmp_r, gmp_r2) != 0) {
			gmp_printf("Modular Subtraction Test failed: %Zd ~= %Zd\n", gmp_r, gmp_r2);
			TEST_FAILURE;
		}

	}
	printf("Modular Subtraction Test Passed.\n");
	TEST_SUCCESS;
}

TEST_MAIN;


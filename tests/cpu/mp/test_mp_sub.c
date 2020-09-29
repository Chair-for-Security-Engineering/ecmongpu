#include <gmp.h>
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"


int test() {
	gmp_randstate_t rand;
	gmp_randinit_default(rand);

	mpz_t gmp_a, gmp_b, gmp_r, gmp_r2;
	mpz_init(gmp_a);
	mpz_init(gmp_b);
	mpz_init(gmp_r);
	mpz_init(gmp_r2);

	mp_t a, b, r;

	for (int i = 0; i < TEST_RUNS; i++) {
		do {
			mpz_urandomb(gmp_a, rand, BITWIDTH);
			mpz_urandomb(gmp_b, rand, BITWIDTH);
		} while (mpz_cmp(gmp_a, gmp_b) <= 0);

		mpz_to_mp(a, gmp_a);
		mpz_to_mp(b, gmp_b);

		mp_sub(r, a, b);
		mpz_sub(gmp_r, gmp_a, gmp_b);

		mp_to_mpz(gmp_r2, r);
		if (mpz_cmp(gmp_r, gmp_r2) != 0) {
			gmp_printf("Subtraction Test failed: %Zd ~= %Zd\n", gmp_r, gmp_r2);
			TEST_FAILURE;
		}
	}
	printf("(Positive) Subtraction Test Passed.\n");
	TEST_SUCCESS;
}

TEST_MAIN;


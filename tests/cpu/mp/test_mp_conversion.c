#include <gmp.h>
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"


int test() {
	gmp_randstate_t rand;
	gmp_randinit_default(rand);

	mpz_t gmp_a, gmp_b;
	mpz_init(gmp_a);
	mpz_init(gmp_b);
	mp_t a;
	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_urandomb(gmp_a, rand, BITWIDTH);
		mpz_to_mp(a, gmp_a);
		mp_to_mpz(gmp_b, a);
		if (mpz_cmp(gmp_a, gmp_b) != 0) {
			gmp_printf("Conversion Test failed: %Zd ~= %Zd\n", gmp_a, gmp_b);
			TEST_FAILURE;
		}
	}
	TEST_SUCCESS;
}

TEST_MAIN;


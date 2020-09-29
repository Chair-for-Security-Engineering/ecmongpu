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

	mp_t a, b;

	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_urandomb(gmp_a, rand, BITWIDTH - 1);
		mpz_urandomb(gmp_b, rand, BITWIDTH - 1);

		mpz_to_mp(a, gmp_a);
		mpz_to_mp(b, gmp_b);

		if (mpz_cmp(gmp_a, gmp_b) != mp_cmp(a, b)) {
			printf("Comparison (cmp) Test failed: %d  %d\n", (mpz_cmp(gmp_a, gmp_b)), mp_cmp(a, b));
			TEST_FAILURE;
		}
	}
	printf("Comparison (cmp) Test Passed.\n");
	TEST_SUCCESS;
}


TEST_MAIN;


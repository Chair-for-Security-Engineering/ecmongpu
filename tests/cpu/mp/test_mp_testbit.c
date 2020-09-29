#include <gmp.h>
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"

int test() {
	gmp_randstate_t rand;
	gmp_randinit_default(rand);

	mpz_t gmp_a;
	mpz_init(gmp_a);

	mp_t a;

	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_urandomb(gmp_a, rand, BITWIDTH);

		mpz_to_mp(a, gmp_a);
		for (int j = 0; j < BITWIDTH; j++) {
			if (mpz_tstbit(gmp_a, j) != (int) mp_test_bit(a, j)) {
				printf("Testbit Test failed on bit %d\n", j);
				TEST_FAILURE;
			}
		}
	}
	printf("Testbit Test Passed.\n");
	TEST_SUCCESS;
}

TEST_MAIN;


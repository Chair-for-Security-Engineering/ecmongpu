#include <gmp.h>
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"

int test() {
	gmp_randstate_t rand;
	gmp_randinit_default(rand);

	mpz_t gmp_a, gmp_r, gmp_r2;
	mpz_init(gmp_a);
	mpz_init(gmp_r);
	mpz_init(gmp_r2);

	mp_t a;

	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_urandomb(gmp_a, rand, BITWIDTH);

		mpz_to_mp(a, gmp_a);
		mp_sr_limbs(a, 1);
		mpz_tdiv_q_2exp(gmp_r, gmp_a, LIMB_BITS);
		mp_to_mpz(gmp_r2, a);
		if (mpz_cmp(gmp_r, gmp_r2) != 0) {
			gmp_printf("sr_limbs Test failed: %Zx ~= %Zx\n", gmp_r, gmp_r2);
			TEST_FAILURE;
		}
	}
	printf("sr_limbs Test Passed.\n");
	TEST_SUCCESS;
}


TEST_MAIN;


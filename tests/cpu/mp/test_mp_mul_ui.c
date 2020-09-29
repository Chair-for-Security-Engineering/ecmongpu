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

	mp_t a, r;

	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_urandomb(gmp_a, rand, BITWIDTH-LIMB_BITS);
		mpz_urandomb(gmp_b, rand, LIMB_BITS);

		mpz_to_mp(a, gmp_a);

		mp_mul_ui(r, a, mpz_get_ui(gmp_b));
		mpz_mul_ui(gmp_r, gmp_a, mpz_get_ui(gmp_b));

		mp_to_mpz(gmp_r2, r);
		if (mpz_cmp(gmp_r, gmp_r2) != 0) {
			gmp_printf("mul_ui Test failed: %Zx ~= %Zx\n", gmp_r, gmp_r2);
			TEST_FAILURE;
		}
	}
	printf("mul_ui Test Passed.\n");
	TEST_SUCCESS;
}

TEST_MAIN;


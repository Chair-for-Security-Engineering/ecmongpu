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
		mpz_urandomb(gmp_b, rand, (BITWIDTH / 2));
		mpz_urandomb(gmp_a, rand, (BITWIDTH / 2));

		mpz_to_mp(a, gmp_a);
		mpz_to_mp(b, gmp_b);
		mp_mul(r, a, b);
		mpz_mul(gmp_r, gmp_a, gmp_b);

		mp_to_mpz(gmp_r2, r);
		if (mpz_cmp(gmp_r, gmp_r2) != 0) {
			gmp_printf("Multiplication Test failed: %Zx ~= %Zx\n", gmp_r, gmp_r2);
			printf("Size of result: %ld", mpz_sizeinbase(gmp_r, 2));
			printf("Size of result: %ld", mpz_sizeinbase(gmp_r2, 2));
			TEST_FAILURE;
		}
	}
	printf("Multiplication Test Passed.\n");
	TEST_SUCCESS;
}


TEST_MAIN;


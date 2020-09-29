#include <gmp.h>
#include <time.h>
#include <stdlib.h>
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"


int test() {
	srand(time(NULL));
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	mpz_t gmp_a, gmp_b, gmp_r, gmp_r2;
	mpz_init(gmp_a);
	mpz_init(gmp_b);
	mpz_init(gmp_r);
	mpz_init(gmp_r2);

	mp_t a, b, r;

	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_urandomb(gmp_a, gmprand, BITWIDTH - 1);
		mpz_urandomb(gmp_b, gmprand, BITWIDTH - 1);

		mpz_to_mp(a, gmp_a);
		mpz_to_mp(b, gmp_b);

		mp_add(r, a, b);
		mpz_add(gmp_r, gmp_a, gmp_b);

		mp_to_mpz(gmp_r2, r);
		if (mpz_cmp(gmp_r, gmp_r2) != 0) {
			gmp_printf("Addition Test failed: %Zd ~= %Zd\n", gmp_r, gmp_r2);
			gmp_printf("Addition Test failed: %Zx ~= %Zx\n", gmp_r, gmp_r2);
			printf("Size of result: %ld", mpz_sizeinbase(gmp_r, 2));
			printf("Size of result: %ld", mpz_sizeinbase(gmp_r2, 2));
			TEST_FAILURE;
		} else {
			printf(".");
		}
	}
	printf("Addition Test Passed.\n");
	TEST_SUCCESS;
}


TEST_MAIN;


#include "ecc/naf.h"
#include "test/testutil.h"

int test() {
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_t gmp_a, gmp_a2;
		mpz_init(gmp_a);
		mpz_init(gmp_a2);

		mpz_urandomb(gmp_a, gmprand, 2 * BITWIDTH);

		size_t naf_size = mpz_sizeinbase(gmp_a, 2) + 64;
		naf_t naf = (naf_t) malloc(naf_size * sizeof(int8_t));
		for (int win = 2; win <= 4; win++) {
			int naf_digits = to_naf(naf, naf_size, gmp_a, win);
			printf("NAF digits: %d\n", naf_digits);
			if (naf_digits < 0) {
				TEST_FAILURE;
			}
			from_naf(gmp_a2, naf, naf_digits);

			print_naf(naf, naf_digits);
			if (mpz_cmp(gmp_a, gmp_a2) != 0) {
				gmp_printf("orig: %Zi\n", gmp_a);
				gmp_printf("conv: %Zi\n", gmp_a2);
				printf("NAF and back failed\n");
				TEST_FAILURE;
			}
		}
	}

	TEST_SUCCESS;
}

TEST_MAIN;


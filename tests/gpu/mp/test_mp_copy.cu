#include <gmp.h>
#include <cuda.h>
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"

int test() {
	gmp_randstate_t rand;
	gmp_randinit_default(rand);

	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_t gmp_a;
		mpz_init(gmp_a);
		mpz_urandomb(gmp_a, rand, BITWIDTH);
		mp_t a, b;
		mpz_to_mp(a, gmp_a);

		mp_p a_dev;
		mp_dev_init(&a_dev);

		mp_copy_to_dev(a_dev, a);
		mp_copy_from_dev(b, a_dev);

		if (mp_cmp(a, b) != 0) {
			TEST_FAILURE;
		}
	}
	printf("Cuda copy mp passed.\n");
	TEST_SUCCESS;
}


TEST_MAIN;


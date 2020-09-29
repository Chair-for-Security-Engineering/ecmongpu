#include <gmp.h>
#include <cuda.h>
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"


__global__
void cuda_mp_gt(mp_t r, const mp_t a, const mp_t b) {
	r[0] = mp_gt(a, b);
	return;
}

__global__
void cuda_mp_print(mp_p dev_a) {
	mp_print(dev_a);
}

int test() {
	gmp_randstate_t rand;
	gmp_randinit_default(rand);

	mpz_t gmp_a, gmp_b, gmp_r, gmp_r2;
	mpz_init(gmp_a);
	mpz_init(gmp_b);
	mpz_init(gmp_r);
	mpz_init(gmp_r2);

	mp_t a, b, r;

	mp_p dev_a;
	mp_dev_init(&dev_a);
	mp_p dev_b;
	mp_dev_init(&dev_b);
	mp_p dev_r;
	mp_dev_init(&dev_r);


	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_urandomb(gmp_a, rand, BITWIDTH - 1);
		mpz_urandomb(gmp_b, rand, BITWIDTH - 1);

		mpz_to_mp(a, gmp_a);
		mpz_to_mp(b, gmp_b);

		mp_copy_to_dev(dev_a, a);
		mp_copy_to_dev(dev_b, b);

#ifdef LOG_LEVEL_VERBOSE_ENABLED
		printf("host:\n");
		mp_print(a);
		mp_print(b);

		printf("device:\n");
		cuda_mp_print<<<1,1>>>(dev_a);
		cuda_mp_print<<<1,1>>>(dev_b);
#endif

		cuda_mp_gt << < 1, 1 >> > (dev_r, dev_a, dev_b);

		mp_copy_from_dev(r, dev_r);

		if ((mpz_cmp(gmp_a, gmp_b) > 0) != r[0]) {
			printf("Comparison (gt) Test failed: %d  %d\n", (mpz_cmp(gmp_a, gmp_b)), r[0]);
			TEST_FAILURE;
		}
	}
	printf("Comparison (gt) Test Passed.\n");
	TEST_SUCCESS;
}

TEST_MAIN;

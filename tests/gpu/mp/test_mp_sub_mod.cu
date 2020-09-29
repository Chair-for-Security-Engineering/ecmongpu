#include <gmp.h>
#include <cuda.h>
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"


__global__
void cuda_mp_sub_mod(mp_t r, const mp_t a, const mp_t b, const mp_t n) {
	mp_sub_mod(r, a, b, n);
	return;
}


__global__
void cuda_mp_print(mp_p dev_a) {
	mp_print(dev_a);
}

int test() {
	gmp_randstate_t rand;
	gmp_randinit_default(rand);

	mpz_t gmp_a, gmp_b, gmp_r, gmp_r2, gmp_n;
	mpz_init(gmp_a);
	mpz_init(gmp_b);
	mpz_init(gmp_r);
	mpz_init(gmp_n);
	mpz_init(gmp_r2);

	mp_t a, b, r, n;

	mp_p dev_a;
	mp_dev_init(&dev_a);
	mp_p dev_b;
	mp_dev_init(&dev_b);
	mp_p dev_r;
	mp_dev_init(&dev_r);
	mp_p dev_n;
	mp_dev_init(&dev_n);


	for (int i = 0; i < TEST_RUNS; i++) {
		mpz_urandomb(gmp_n, rand, BITWIDTH);
		mpz_nextprime(gmp_n, gmp_n);

	  mpz_urandomb(gmp_a, rand, BITWIDTH);
		mpz_urandomb(gmp_b, rand, BITWIDTH);

		mpz_mod(gmp_a, gmp_a, gmp_n);
		mpz_mod(gmp_b, gmp_b, gmp_n);

		mpz_to_mp(a, gmp_a);
		mpz_to_mp(b, gmp_b);
		mpz_to_mp(n, gmp_n);

		mp_copy_to_dev(dev_a, a);
		mp_copy_to_dev(dev_n, n);
		mp_copy_to_dev(dev_b, b);


#ifdef LOG_LEVEL_VERBOSE_ENABLED
		printf("host:\n");
		mp_print(a);
		mp_print(b);
		mp_print(n);

		printf("device:\n");
		cuda_mp_print<<<1,1>>>(dev_a);
		cuda_mp_print<<<1,1>>>(dev_b);
		cuda_mp_print<<<1,1>>>(dev_n);
#endif

		cuda_mp_sub_mod << < 1, 1 >> > (dev_r, dev_a, dev_b, dev_n);

		mp_copy_from_dev(r, dev_r);

		mpz_sub(gmp_r, gmp_a, gmp_b);
		mpz_mod(gmp_r, gmp_r, gmp_n);

#ifdef LOG_LEVEL_VERBOSE_ENABLED
		gmp_printf("N:\t%Zi\nA:\t%Zi\nB:\t%Zi\n", gmp_n, gmp_a, gmp_b);
		gmp_printf("A-B:\t%Zi\n", gmp_r);
#endif

		mp_to_mpz(gmp_r2, r);
		if (mpz_cmp(gmp_r, gmp_r2) != 0) {
			gmp_printf("Modular Subtraction Test failed: %Zd ~= %Zd\n", gmp_r, gmp_r2);
			TEST_FAILURE;
		}
	}
	printf("Modular Subtraction Test Passed.\n");
	TEST_SUCCESS;
}

TEST_MAIN;


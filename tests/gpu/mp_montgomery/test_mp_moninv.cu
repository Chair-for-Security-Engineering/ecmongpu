#include <gmp.h>
#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"
#include <time.h>

__global__
void cuda_mon_inv(mp_t r, const mp_t a, const mon_info *info) {
	mon_inv(r, a, info);
}

int test() {
  //srand(time(NULL));
  srand(1);
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	mpz_t gmp_a, gmp_ainv, gmp_n;
	mpz_init(gmp_a);
	mpz_init(gmp_ainv);
	mpz_init(gmp_n);
	
	mp_t a, ainv, n;

	for (int i = 0; i < 100 * TEST_RUNS; i++) {

		mpz_urandomb(gmp_n, gmprand, BITWIDTH);
		/* n has to be an odd modulus */
		mpz_setbit(gmp_n, 0);
		mpz_urandomb(gmp_a, gmprand, BITWIDTH);
		mpz_nextprime(gmp_n, gmp_n);
		mpz_mod(gmp_a, gmp_a, gmp_n);

		mpz_to_mp(n, gmp_n);

		/* Allocate and calculate Montgomery Constants for this modulus */
		mon_info info;
		mon_info_populate(n, &info);

		mpz_to_mp(a, gmp_a);

    /* Compute gmp's inverse */
    mpz_invert(gmp_a, gmp_a, gmp_n);
    gmp_printf("Inverse: %Zi\n", gmp_a);

    /* compute mp's invers */
		to_mon(a, a, &info);

    mp_p dev_a, dev_r;
    mp_dev_init(&dev_a);
    mp_dev_init(&dev_r);

    mp_copy_to_dev(dev_a, a);

    mon_info *dev_info = mon_info_copy_to_dev(&info);

    cuda_mon_inv<<<1,1>>>(dev_r, dev_a, dev_info);
    
    cudaDeviceSynchronize();

    mp_copy_from_dev(ainv, dev_r);

		from_mon(ainv, ainv, &info);
		mp_to_mpz(gmp_ainv, ainv);

		mpz_mod(gmp_ainv, gmp_ainv, gmp_n);

		if (mpz_cmp(gmp_a, gmp_ainv) != 0) {
			printf("Inversion test failed\n");
			gmp_printf("GMP:\t%Zi\nMP:\t%Zi\n", gmp_a, gmp_ainv);
			TEST_FAILURE;
		} else {
			printf("Inversion test success\n");
		}

	}
	mpz_clear(gmp_a);
	mpz_clear(gmp_ainv);
	mpz_clear(gmp_n);
	gmp_randclear(gmprand);

	TEST_SUCCESS;
}

TEST_MAIN;


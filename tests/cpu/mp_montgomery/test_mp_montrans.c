#include <gmp.h>
#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "test/testutil.h"


int test() {
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	for (int i = 0; i < TEST_RUNS; i++) {

		mpz_t gmp_a, gmp_a2, gmp_n;
		mpz_init(gmp_a);
		mpz_init(gmp_a2);
		mpz_init(gmp_n);

		mpz_urandomb(gmp_n, gmprand, BITWIDTH);
		/* n has to be an odd modulus */
		mpz_setbit(gmp_n, 0);
		mpz_urandomb(gmp_a, gmprand, BITWIDTH);
		mpz_mod(gmp_a, gmp_a, gmp_n);


		mp_t a, a2, n;

		mpz_to_mp(n, gmp_n);

		/* Allocate and calculate Montgomery Constants for this modulus */
		mon_info info;
		mon_info_populate(n, &info);

		mpz_to_mp(a, gmp_a);

		to_mon(a, a, &info);

		from_mon(a2, a, &info);
		mp_to_mpz(gmp_a2, a2);

		if (mpz_cmp(gmp_a, gmp_a2) != 0) {
			printf("Transform test failed\n");
			gmp_printf("before:\t%Zi\nafter:\t%Zi\n", gmp_a, gmp_a2);
			TEST_FAILURE;
		}

		mpz_clear(gmp_a);
		mpz_clear(gmp_a2);
		mpz_clear(gmp_n);
	}
	gmp_randclear(gmprand);

	TEST_SUCCESS;
}

TEST_MAIN;


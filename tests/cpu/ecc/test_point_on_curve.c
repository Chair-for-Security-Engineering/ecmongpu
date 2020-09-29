#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "ecc/twisted_edwards.h"
#include "test/testutil.h"

#include "log.h"


int test() {
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	mpz_t gmp_n;
	mp_t n;

	/* n is prime. all curve arithmetic should stay on curve */
	mpz_init(gmp_n);
	mpz_urandomb(gmp_n, gmprand, BITWIDTH);
	mpz_nextprime(gmp_n, gmp_n);

	mpz_to_mp(n, gmp_n);

	point_tw_ed p1;
	curve_tw_ed curve;

	mon_info info;
	mon_info_populate(n, &info);

	point_gkl2016 phelper;
  pthread_mutex_init(&phelper.mutex, NULL);
	mpz_init_set_ui(phelper.x, 5);
	mpz_init_set_ui(phelper.y, 8);

	for (int curve_gen = 0; curve_gen < job_generators_len; curve_gen++) {
		LOG_DEBUG("Using curve generator %d: %s", curve_gen, job_generators_names[curve_gen]);

		for (int i = 0; i < TEST_RUNS; i++) {

			(job_generators[curve_gen])(&p1, &curve, &info, gmprand, (void*)&phelper);
			int cmp = tw_ed_point_on_curve(&p1, &curve, &info);
			printf("point on curve: %d\n", cmp);

			if (!cmp) {
				tw_ed_print_point(&p1, &info);
				printf("Error: p1 not on curve\n");
				TEST_FAILURE;
			}

			mp_add_ui(p1.x, p1.x, 1);

			cmp = tw_ed_point_on_curve(&p1, &curve, &info);
			printf("point on curve: %d\n", cmp);
			if (cmp) {
				printf("Error: p should not be on curve\n");
				TEST_FAILURE;
			}

		}
	}

	/* Cleanup GMP */
	mpz_clear(gmp_n);
	gmp_randclear(gmprand);
	TEST_SUCCESS;
}

TEST_MAIN;


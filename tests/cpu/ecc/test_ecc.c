
#include "mp/mp_montgomery.h"
#include "mp/mp.h"
#include "mp/gmp_conversion.h"
#include "log.h"
#include "ecc/twisted_edwards.h"
#include "test/testutil.h"


int test() {
	srand(time(NULL));
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	mpz_t gmp_n;
	mp_t n;

	/* n is prime. all curve arithmetic should stay on curve */
	mpz_init(gmp_n);

	point_tw_ed p1, ptmp1, ptmp2;
	point_tw_ed padd;
	curve_tw_ed curve;
	mon_info info;

	point_gkl2016 phelper;
  pthread_mutex_init(&phelper.mutex, NULL);
	mpz_init_set_ui(phelper.x, 5);
	mpz_init_set_ui(phelper.y, 8);

	for (int curve_gen = 0; curve_gen < job_generators_len; curve_gen++) {
		LOG_DEBUG("Using curve generator %d: %s", curve_gen, job_generators_names[curve_gen]);

		for (int i = 0; i < TEST_RUNS; i++) {
			mpz_urandomb(gmp_n, gmprand, BITWIDTH);
			mpz_nextprime(gmp_n, gmp_n);

			mpz_to_mp(n, gmp_n);

			mon_info_populate(n, &info);

			(job_generators[curve_gen])(&p1, &curve, &info, gmprand, (void*)&phelper);

			tw_ed_copy_point(&padd, &p1);

			if (!tw_ed_point_on_curve(&p1, &curve, &info)) {
				tw_ed_print_point(&p1, &info);
				printf("Error: p1 not on curve\n");
				TEST_FAILURE;
			}

			// Double
			for (int j = 0; j < 1000; j++) {
				tw_ed_double(&p1, &p1, &curve, &info, true);
				tw_ed_add(&padd, &padd, &padd, &curve, &info, true);

				tw_ed_scale_point(&ptmp1, &p1, &info);
				tw_ed_scale_point(&ptmp2, &padd, &info);

				if (mp_cmp(ptmp1.x, ptmp2.x) != 0 || mp_cmp(ptmp1.y, ptmp2.y) != 0 || mp_cmp(ptmp1.z, ptmp2.z) != 0) {
					printf("Addition and doubling does not match!");
					TEST_FAILURE;
				}
			}
			// Triple
			for (int j = 0; j < 1000; j++) {
				tw_ed_triple(&p1, &p1, &curve, &info, true);
				point_tw_ed ptmp;
				tw_ed_add(&ptmp, &padd, &padd, &curve, &info, true);
				tw_ed_add(&padd, &ptmp, &padd, &curve, &info, true);

				tw_ed_scale_point(&ptmp1, &p1, &info);
				tw_ed_scale_point(&ptmp2, &padd, &info);

				if (mp_cmp(ptmp1.x, ptmp2.x) != 0 || mp_cmp(ptmp1.y, ptmp2.y) != 0 || mp_cmp(ptmp1.z, ptmp2.z) != 0) {
					printf("Addition and tripling does not match!");
					TEST_FAILURE;
				}
			}

			if (!tw_ed_point_on_curve(&p1, &curve, &info)) {
				print_info(&info);
				printf("p2 in test run %d:\n", i);
				tw_ed_print_point(&p1, &info);
				tw_ed_print_curve(&curve);
				printf("Error: p2 not on curve\n");
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


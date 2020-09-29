#include "test/testutil.h"
#include "ecm/factor_task.h"

int is_sorted(factor_list list) {
	for (factor_list f = list; f != NULL; f = f->next) {
		if (f->next != NULL && mpz_cmp(f->next->factor, f->factor) < 0) {
			return 0; // Failure
		}
	}
	return 1;
}

int print_list(factor_list list) {
	int i = 0;
	for (factor_list f = list; f != NULL; f = f->next) {
		gmp_printf("%d: %Zi\n", i, f->factor);
		i++;
	}
	return 1;
}

int test() {
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	gmp_randseed_ui(gmprand, rand());

	mpz_t tmp;
	mpz_init(tmp);

	for (int i = 0; i < TEST_RUNS; i++) {

		factor_list l = factor_list_new();

		/* Fill list randomly */
		for (int j = 0; j < 10; j++) {
			mpz_urandomb(tmp, gmprand, BITWIDTH);
			factor_list_push(&l, tmp);
		}

		print_list(l);

		factor_list_sort(&l);

		print_list(l);


		if (!is_sorted(l)) {
			TEST_FAILURE;
		}
	}

	/* Cleanup GMP */
	mpz_clear(tmp);
	gmp_randclear(gmprand);

	TEST_SUCCESS;
}

TEST_MAIN;


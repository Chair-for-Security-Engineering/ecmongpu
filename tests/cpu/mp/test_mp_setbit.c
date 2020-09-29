#include <gmp.h>
#include "mp/mp.h"
#include "test/testutil.h"

int test() {
	mp_t a;

	for (int i = 2; i < BITWIDTH; i++) {
		mp_set_ui(a, 0);
		mp_set_bit(a, i);

		if (mp_test_bit(a, i - 1) != 0) {
			printf("Setbit Test failed on bit %d\n", i - 1);
			TEST_FAILURE;
		}

		if (mp_test_bit(a, i) != 1) {
			printf("Setbit Test failed on bit %d\n", i);
			TEST_FAILURE;
		}
	}
	printf("Setbit Test Passed.\n");
	TEST_SUCCESS;
}


TEST_MAIN;


/*
	co-ecm
	Copyright (C) 2018  Jonas Wloka

	This program is free software: you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
			the Free Software Foundation, either version 3 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful,
			but WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	GNU General Public License for more details.

	You should have received a copy of the GNU General Public License
	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef CO_ECM_TESTUTIL_H
#define CO_ECM_TESTUTIL_H

/**
 * Macro to print out PASSED. Used with CTest.
 */
#define TEST_SUCCESS printf("\nPASSED\n"); \
                     return 0;

/**
 * Macro to print out FAILURE. Used with CTest.
 */
#define TEST_FAILURE printf("\nFAILURE\n"); \
                     return -1;

/**
 * Default number of runs for each (randomized) test.
 */
#define TEST_RUNS 100

/**
 * Convenience macro used in all tests.
 */
#define TEST_MAIN int main(){ return test(); }

#endif //CO_ECM_TESTUTIL_H

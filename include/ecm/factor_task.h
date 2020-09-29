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

#ifndef CO_ECM_FACTOR_JOB_H
#define CO_ECM_FACTOR_JOB_H

typedef struct _factor_task *factor_task;

#include <gmp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdbool.h>
#include <pthread.h>

#include "ecc/naf.h"
#include "ecc/twisted_edwards.h"
#include "ecm/batch.h"
#include "config/config.h"

#ifdef __cplusplus
extern "C" {
#endif

extern pthread_mutex_t mutex_factor_list;
extern pthread_mutex_t mutex_factor_tree;
extern pthread_mutex_t mutex_factor_queue;


/**
 * Linked list of factors.
 */
typedef struct _factor_list {
	mpz_t factor; /**< Factor */
	int exponent; /**< Power of the factor */
	struct _factor_list *next;
} *factor_list;

/**
 * Task and parameters for a single number to be factorized.
 */
struct _factor_task {
	unsigned int id;
	mpz_t n; 				/**< Number to be factorized */
	mpz_t composite;		/**< Working copy of the number to be factorized. Decreases with known factors. */
	factor_list factors;	/**< List of already known factors */

	int effort_allocated;		/**< Effort allocated to this task (not yet executed). */
	int effort_spent;		/**< Effort already spent on factoring. */

	int priority;		/**< Effort already spent on factoring. */

	run_config config;		/**< Pointer to run_config associated with this task */

	bool done;				/**< Whether all factors were found and this task is done. */

	pthread_mutex_t *mutex;

	point_gkl2016 p_gkl2016;

};

/**
 * Add a factor to a list of factors.
 *
 * Pushes in front of the list. Returns new list head.
 * @param head		Head of the list to add factor to. Updated to new head after.
 * @param factor	Factor to add to the list.
 * @return			New list head after insertion.
 */
factor_list factor_list_push(factor_list *head, mpz_t factor);

/**
 * Deletes a factor task and frees its memory
 *
 * @param task 		Task to delete
 * @param config 	Runtime configuration
 */
void factor_task_unregister(factor_task task, run_config config);

/**
 * Add a factor to a list of factors, checking for already existing factors.
 *
 * If a factor already exits, the new factor is discarded.
 * For each already known factor, the gcd(old_factor, new_factor) is computed and added to the list if non-trivial. Both
 * already known factors are divided by the gcd and the results itself are added to the list via a recursive call to
 * this function.
 *
 * Pushes in front of the list. Returns new list head.
 *
 * @param head		Head of the list to add factor to. Updated to new head after.
 * @param factor	Factor to add to the list.
 * @return			New list head after insertion.
 */
factor_list factor_list_push_unique(factor_list *head, mpz_t factor);

/**
 * Removes a factor from the list of factors.
 *
 * Checks for equality (i.e. via mpz_cmp()) of the factor, not the exponent or any memory adresses.
 *
 * @param factors 	List of factors.
 * @param factor	Factor to remove.
 * @return			New list head after removal.
 */
int factor_list_remove(factor_list *factors, mpz_t factor);

/**
 * New factor list
 *
 * @return Pointer to list head
 */
factor_list factor_list_new();

/**
 * Sort a list of factors in ascending order in place.
 *
 * As of now, this does use bubble sort :)
 *
 * @param list
 */
void factor_list_sort(factor_list *list);

/**
 * Search through list and remove all duplicate factors.
 * @param list 	List to sort
 */
void factor_list_remove_duplicates(factor_list *list);

/**
 * Create a new factor task with given parameters.
 *
 * @param n				Number to be factored.
 * @param config		Config this task is asscociated to
 * @return				Factor initialized with given parameters.
 */
factor_task factor_task_new(unsigned int id, mpz_t n, run_config config);

/**
 * Comparison function in use for tree priority queue
 *
 * @param left 		First node to compare (factor_task)
 * @param right		Second node to compare (factor_task)
 * @return
 */
int task_tree_cmp(const void *left, const void *right);

/**
 * Free memory occupied by factor_task
 * @param task freed task
 */
void factor_task_free(factor_task task);

/**
 * Log factor_task to DEBUG log level.
 *
 * @param task to log.
 */
void factor_task_log(factor_task task);

/**
 * Get the next (not already finished) factor task from the queue
 *
 * @param config 	Runtime configuration
 * @return 			A not yet finished factor task to work on
 */
factor_task factor_task_get_next(run_config config);

/**
 * Add a factor_task to the work queue
 *
 * @param task 		Task to add
 * @param config 	Runtime configuration
 */
void factor_task_enqueue(factor_task task, run_config config);

/**
 * Check if the task with the given id has been successfully worked on, prints to the configured output
 * and removes any memory asscociated.
 *
 * @param task_id ID of the task to finalize
 * @param config Runtime configuration
 */
void task_finish(int task_id, run_config config);

/**
 * Check (and add on success) a given factor candidate (in Montgomery domain) for the task
 *
 * @param task_id	Task ID of the corresponding task
 * @param config	Runtime configuration
 * @param a			Possible factor
 * @param info		Montgomery domain information
 */
void task_add_factor_mp_mon(int task_id, run_config config, mp_t a, mon_info *info);

/**
 * Check (and add on success) a given factor candidate for the task
 *
 * @param task_id	Task ID of the corresponding task
 * @param config	Runtime configuration
 * @param a			Possible factor
 */
void task_add_factor(int task_id, run_config config, mp_t a);

/**
 * Increase the spent effort for a task
 *
 * @param config 	Runtime configuration
 * @param task_id 	Task ID
 */
void factor_task_inc_effort(int task_id, run_config config);


/**
 * Check for correctness of factors saved with a task
 * @param task
 */
void ecm_check_factors(factor_task task);

/**
 * Return a task by its ID
 * @param id		Task ID
 * @param config	Runtime configuration
 * @return 			Task if found, NULL otherwise
 */
factor_task factor_task_by_id_lock(unsigned int id, run_config config);

/**
 * Reduce the still-to-factor composite of a task by the factors already found
 * @param task
 */
void ecm_divide_out(factor_task task);

/**
 * Type definition for functions returning whether or not a task is 'done'/
 */
typedef int (*ecm_done_fn)(factor_task task);

/**
 * Funtion returning a positive result if all prime factors of the task have been found
 * @param task
 * @return
 */
int ecm_fully_done(factor_task task);

/**
 * Function returning positive result if any prime factor of the task has been found
 * @param task
 * @return
 */
int ecm_factor_found_done(factor_task task);

#ifdef __cplusplus
};
#endif

#endif //CO_ECM_FACTOR_JOB_H

#include <stdlib.h>
#include <search.h>
#include "ecm/factor_task.h"
#include "log.h"
#include "cudautil.h"

pthread_mutex_t mutex_factor_list = PTHREAD_MUTEX_INITIALIZER;
pthread_rwlock_t rwlock_factor_tree = PTHREAD_RWLOCK_INITIALIZER;


factor_task _factor_task_by_id_unsafe(unsigned int id, run_config config) {
	struct _factor_task task;
	task.id = id;

	factor_task *task_found = (factor_task *) tfind(&task, &config->task_tree_root, task_tree_cmp);

	if(task_found) {
		return *task_found;
	} else {
		LOG_VERBOSE("Task lookup, not found: %d", id);
		return NULL;
	}
}

factor_list factor_list_push(factor_list *head, mpz_t factor) {
	// Allocate memory for node
	factor_list new_node = (struct _factor_list *) malloc(sizeof(struct _factor_list));
	LOG_VERBOSE("Adding factor %Zi to list", factor);

	mpz_init_set(new_node->factor, factor);
	new_node->exponent = 0;
	new_node->next = (*head);

	// Change head pointer as new node is added at the beginning
	(*head) = new_node;

	return (*head);
}


factor_list factor_list_push_unique(factor_list *head, mpz_t factor) {
	/* One is not a valid factor */
	if (mpz_cmp_ui(factor, 1) == 0) return (*head);

	int new_factor = 1;
	for (factor_list f = (*head); f != NULL; f = f->next) {
		/* Factor already in list */
		if (mpz_cmp(f->factor, factor) == 0) break;

		/* Compute gcd between new and known factor */
		mpz_t gcd, tmp, old;
		mpz_init(gcd);
		mpz_init(tmp);
		mpz_init(old);

		mpz_gcd(gcd, factor, f->factor);
		/* If gcd is larger, remove old factor, add gcd to list, and divide old and new factor by gcd and add to list */
		if (mpz_cmp_ui(gcd, 1) != 0) {
			/* Add gcd */
			factor_list_push_unique(head, gcd);

			/* Copy old factor) */
			mpz_set(old, f->factor);
			/* Remove old_factor */
			factor_list_remove(head, f->factor);

			/* Add old_factor/gcd */
			mpz_divexact(tmp, old, gcd);
			if (mpz_cmp_ui(tmp, 1) != 0)
				factor_list_push_unique(head, tmp);

			/* Add new_factor/gcd */
			mpz_divexact(tmp, factor, gcd);
			if (mpz_cmp_ui(tmp, 1) != 0)
				factor_list_push_unique(head, tmp);

			new_factor = 0;
		}
		mpz_clear(gcd);
		mpz_clear(tmp);
		mpz_clear(old);
		if (new_factor == 0) break;
	}
	if (new_factor) factor_list_push(head, factor);

	return (*head);
}


factor_list factor_list_new() {
	return NULL;
}

void swap(factor_list a, factor_list b) {
	mpz_t tmp_factor;
	mpz_init_set(tmp_factor, a->factor);

	int tmp_exponent = a->exponent;

	mpz_set(a->factor, b->factor);
	a->exponent = b->exponent;

	mpz_set(b->factor, tmp_factor);
	b->exponent = tmp_exponent;

	mpz_clear(tmp_factor);
}

void factor_list_bubblesort(factor_list *head) {
	/* Abort on empty list */
	if ((*head) == NULL)
		return;

	int swapped;
	factor_list ptr1;
	factor_list lptr = NULL;
	do {
		swapped = 0;
		ptr1 = (*head);

		while (ptr1->next != lptr) {
			if (mpz_cmp(ptr1->factor, ptr1->next->factor) > 0) {
				swap(ptr1, ptr1->next);
				swapped = 1;
			}
			ptr1 = ptr1->next;
		}
		lptr = ptr1;
	} while (swapped);
}

void factor_list_sort(factor_list *head) {
	factor_list_bubblesort(head);
}


void factor_list_remove_duplicates(factor_list *head) {
	/* Empty list */
	if (*head == NULL) return;
	factor_list_sort(head);

	for (factor_list f = (*head); f != NULL; f = f->next) {
		if (f->next && mpz_cmp(f->factor, f->next->factor) == 0) {
			factor_list tmp = f->next;
			/* Skip over */
			f->next = f->next->next;
			mpz_clear(tmp->factor);
			free(tmp);
		}
	}
}


int factor_list_remove(factor_list *factors, mpz_t factor) {
	int ret = 0;
	/* Check for first element */
	if (mpz_cmp((*factors)->factor, factor) == 0) {
		mpz_clear((*factors)->factor);
		factor_list old_head = (*factors);
		(*factors) = (*factors)->next;
		free(old_head);
		return 1;
	}
	for (factor_list f = (*factors); f != NULL; f = f->next) {
		if (f->next != NULL && mpz_cmp(f->next->factor, factor) == 0) {
			factor_list rem = f->next;
			mpz_clear(rem->factor);
			f->next = rem->next;
			free(rem);
			return 1;
		}
	}
	return ret;
}

void task_tree_walk(const void *node, const VISIT which, const int depth) {
	LOG_WARNING("depth: %d", depth);
	factor_task t = *((factor_task *) node);
	factor_task_log(t);
}


factor_task factor_task_by_id_lock(unsigned int task_id, run_config config) {

	factor_task task;

	while(true){
		// Lock tree
		pthread_rwlock_rdlock(&rwlock_factor_tree);
		task = _factor_task_by_id_unsafe(task_id, config);
		// If no task, unlock tree and return
		if(!task){
			pthread_rwlock_unlock(&rwlock_factor_tree);
			LOG_VERBOSE("Task lookup, not found: %d", id);
			return NULL;
		}
		// if task exists, try to acquire task lock
		if(pthread_mutex_trylock(task->mutex) == 0) {
			break;
		}
		// if no task lock, unlock tree and retry
		pthread_rwlock_unlock(&rwlock_factor_tree);
	}

	// task lock acquired
	// release tree lock
	pthread_rwlock_unlock(&rwlock_factor_tree);

	return task;
}



factor_task factor_task_get_next(run_config config) {
	/* Keep popping tasks from queue until a good one is found or none left */
	factor_task task = NULL;

	int priority;

	while ((task = factor_task_pop_priority(config->factor_tasks_queue, &priority)) != NULL) {
		/* Cleanup queues if task done or to abandon */
		if (task->done) {
			LOG_VERBOSE("Popped done task.");
			task_finish(task->id, config);
		} else {
			LOG_VERBOSE("Popped good task.");

#ifdef LOG_VERBOSE_ENABLED
			if(task->id ==1){
				LOG_VERBOSE("Task ID 1\n\tn: %Zi\np=(%Zi,%Zi)", task->n, task->p_gkl2016.x, task->p_gkl2016.y);
			}
#endif

			LOG_VERBOSE("\tInserting with priority %i", task->priority);
			int alloc = __sync_add_and_fetch(&task->effort_allocated, 1);
			if (alloc < config->effort_max) {
				factor_task_enqueue(task, config);
			}
			/* Good task found, break from while */
			break;

		}
	}
	return task;
}

void factor_task_unregister(factor_task task, run_config config) {
	// remove from tree
	factor_task *task_found = (factor_task *) tdelete(task, &config->task_tree_root, task_tree_cmp);

	if(!task_found) {
		LOG_FATAL("Task to unregister not found");
	}
}


void factor_task_inc_effort(int task_id, run_config config){

	factor_task task;

	while(true){
		// Lock tree
		pthread_rwlock_rdlock(&rwlock_factor_tree);
		task = _factor_task_by_id_unsafe(task_id, config);
		// If no task, unlock tree and return
		if(!task){
			pthread_rwlock_unlock(&rwlock_factor_tree);
			return;
		}
		// if task exists, try to acquire task lock
		if(pthread_mutex_trylock(task->mutex) == 0) {
			break;
		}
		// if no task lock, unlock tree and retry
		pthread_rwlock_unlock(&rwlock_factor_tree);
	}

	// task lock acquired
	// release tree lock
	pthread_rwlock_unlock(&rwlock_factor_tree);

	// increase effort
	task->effort_spent++;

	// unlock task
	pthread_mutex_unlock(task->mutex);
}

factor_task factor_task_new(unsigned int task_id, mpz_t n, run_config config) {
	factor_task task = (struct _factor_task *) malloc(sizeof(struct _factor_task));

	task->mutex = (pthread_mutex_t *)malloc(sizeof(pthread_mutex_t));

	pthread_mutex_init(task->mutex, NULL);

	mpz_init_set(task->n, n);
	mpz_init_set(task->composite, n);
	task->id = task_id;
	task->factors = factor_list_new();
	task->effort_spent = 0;
	task->effort_allocated = 0;

	task->done = false;

	task->priority = 10;

	task->config = config;

	// Init task curve construction lock
  	pthread_mutex_init(&task->p_gkl2016.mutex, NULL);
	// acquire tree lock
	pthread_rwlock_wrlock(&rwlock_factor_tree);

	// Remove possible existing
	tdelete(task, &config->task_tree_root, task_tree_cmp);
	// insert into tree
	if(tsearch(task, &config->task_tree_root, task_tree_cmp) == NULL){
		LOG_FATAL("Could not register task %d", task_id);
	};
	pthread_rwlock_unlock(&rwlock_factor_tree);

	mpz_init_set_ui(task->p_gkl2016.x, 5);
	mpz_init_set_ui(task->p_gkl2016.y, 8);
	return task;
}

void factor_task_enqueue(factor_task task, run_config config) {
	factor_task_push(config->factor_tasks_queue, task->priority, task);
}

int task_tree_cmp(const void *left, const void *right) {
	factor_task l = (factor_task) left;
	factor_task r = (factor_task) right;
	return (l->id == r->id) ? 0 : ((l->id < r->id) ? -1 : 1);
}


// Todo: Make log level variable;
void factor_task_log(factor_task task) {
	LOG_DEBUG("Factor task [%p]", task);
	LOG_DEBUG("\tn:\t%Zi", task->n);
	LOG_DEBUG("\tcomposite:\t%Zi", task->composite);
	LOG_DEBUG("\teffort:\t%u", task->effort_spent);
	LOG_DEBUG("\tFactors:");
	for (factor_list f = task->factors; f != NULL; f = f->next) {
		LOG_DEBUG("\t\t%Zi^(%u), ", f->factor, f->exponent);
	}
}

void task_finish(int task_id, run_config config) {
	// acquire tree and task locks
	factor_task task;
	while(true){
		// Lock tree
		pthread_rwlock_rdlock(&rwlock_factor_tree);
		task = _factor_task_by_id_unsafe(task_id, config);
		// If no task, unlock tree and return
		if(!task){
			pthread_rwlock_unlock(&rwlock_factor_tree);
			return;
		}
		// if task exists, acquire task lock
		if(pthread_mutex_trylock(task->mutex) == 0) {
			break;
		}
		// unlock, retry
		pthread_rwlock_unlock(&rwlock_factor_tree);
	}

	// task lock acquired
	// release tree lock
	pthread_rwlock_unlock(&rwlock_factor_tree);

	// check if done
	task->done = (config->ecm_done(task) ||
				(task->effort_allocated == task->effort_spent && task->effort_spent >= config->effort_max));

	// print if done
	if (task->done) {
		int outstrlen = 1000;
		char outstr[outstrlen];
		int written = snprintf(outstr, outstrlen, "%d", task->id);
		if (!task->factors) {
			written += snprintf(outstr + written, outstrlen-written, " 1\n");
		} else {
			for (factor_list f = task->factors; f != NULL && f->factor != NULL; f = f->next) {
				written += gmp_snprintf(outstr + written, outstrlen-written, " %Zi", f->factor);
			}
			written += snprintf(outstr + written, outstrlen-written, "\n");
		}
		config->output(outstr, config);

		// Acquire tree lock and delete task from tree
		pthread_rwlock_wrlock(&rwlock_factor_tree);
		factor_task_unregister(task, config);
		pthread_rwlock_unlock(&rwlock_factor_tree);

		// Task removed from tree and was locked, no other thread should
		// get a lock on the task, unlock here for deletion
		pthread_mutex_unlock(task->mutex);

		// Free task memory and destroy lock
		factor_list fnext = NULL;
		for (factor_list f = task->factors; f != NULL; f = fnext) {
			mpz_clear(f->factor);
			fnext = f->next;
			free(f);
		}
		mpz_clear(task->composite);
		mpz_clear(task->n);
		pthread_mutex_destroy(task->mutex);
		free(task->mutex);
		free(task);

	} else {
		// unlock if not done
		pthread_mutex_unlock(task->mutex);
	}

}


bool factor_is_nontrivial(mpz_t gmp_factor, factor_task task) {
	return (mpz_cmp_ui(gmp_factor, 1) != 0 && mpz_cmp(gmp_factor, task->n) != 0);
}


void task_add_factor(int task_id, run_config config, mpz_t gmp_factor) {
	mpz_t gcd;
	mpz_init(gcd);

	factor_task task;

	// acquire task and tree lock
	while(true){
		// Lock tree
		pthread_rwlock_rdlock(&rwlock_factor_tree);
		task = _factor_task_by_id_unsafe(task_id, config);
		// If no task, unlock tree and return
		if(!task){
			pthread_rwlock_unlock(&rwlock_factor_tree);
			return;
		}
		// if task exists, try to acquire task lock
		if(pthread_mutex_trylock(task->mutex) == 0) {
			break;
		}
		// if no task lock, unlock tree and retry
		pthread_rwlock_unlock(&rwlock_factor_tree);
	}

	// task lock acquired
	// release tree lock
	pthread_rwlock_unlock(&rwlock_factor_tree);

	mpz_gcd(gmp_factor, gmp_factor, task->n);

	/* Check if factor is trivial */
	if (factor_is_nontrivial(gmp_factor, task)) {
		/* Check if factor already in list of known factors */
		int new_factor = 1;
		for (factor_list f = task->factors; f != NULL && f->factor != NULL && new_factor == 1; f = f->next) {
			/* if gcd(potential factor, old factor) != 1 we have already had found this factor */
			mpz_gcd(gcd, gmp_factor, f->factor);
			if (mpz_cmp_ui(gcd, 1) != 0) {
				new_factor = 0;
				break;
			}
		}
		if (new_factor == 1) {
			factor_list_push(&task->factors, gmp_factor);
			LOG_DEBUG("Factor of %Zi found: %Zi", task->n, gmp_factor);
			ecm_check_factors(task);
			ecm_divide_out(task);
		}
	}

	// Unlock task lock
	pthread_mutex_unlock(task->mutex);

	mpz_clear(gcd);
}


void task_add_factor_mp_mon(int task_id, run_config config, mp_t a, mon_info *info) {
	mp_t m;
	mpz_t gmp_factor;
	mpz_init(gmp_factor);
	from_mon(m, a, info);
	mp_to_mpz(gmp_factor, m);

	task_add_factor(task_id, config, gmp_factor);

	mpz_clear(gmp_factor);
}


int ecm_fully_done(factor_task task) {
	/* If all factors found return true */
	LOG_VERBOSE("ECM fully_done");
	if (mpz_cmp_ui(task->composite, 1) == 0 || mpz_probab_prime_p(task->composite, 50) > 0) {
		if (mpz_cmp_ui(task->composite, 1) != 0) {
			factor_list_push_unique(&task->factors, task->composite);
			task->factors->exponent = 1;
		}

		LOG_DEBUG("Factors:");
		for (factor_list f = task->factors; f != NULL; f = f->next) {
			LOG_DEBUG("\t%Zi^(%u), ", f->factor, f->exponent);
		}
		return 1;
	}
	return 0;
}


int ecm_factor_found_done(factor_task task) {
	LOG_VERBOSE("ECM factor_found_done");
	/* If any factor found return true */
	if (task->factors != NULL) {
		return 1;
	}
	return 0;
}


void ecm_divide_out(factor_task task) {
	LOG_DEBUG("Reducing composite by known factors...");
	// Reset task->composite to original n
	mpz_set(task->composite, task->n);
	for (factor_list f = task->factors; f != NULL && f->factor != NULL; f = f->next) {
		if (f->exponent != 0) {
			mpz_t tmp;
			mpz_init_set(tmp, f->factor);
			mpz_pow_ui(tmp, tmp, f->exponent);
			LOG_DEBUG("Dividing by");
			LOG_DEBUG("\t%Zi^(%d) = %Zi", f->factor, (unsigned long int) f->exponent, tmp);
			mpz_divexact(task->composite, task->composite, f->factor);
		} else {
			LOG_WARNING("ecm_divide_out with zero exponent factor called.");
		}
	}
}


void ecm_check_factors(factor_task task) {
	mpz_t my_n;
	mpz_init_set(my_n, task->n);

	mpz_t gcd;
	mpz_init(gcd);

	for (factor_list f = task->factors; f != NULL; f = f->next) {
		if (f->exponent == 0) {
			while (mpz_divisible_p(my_n, f->factor)) {
				mpz_divexact(my_n, my_n, f->factor);
				f->exponent++;
			}
		}
	}

	mpz_clear(my_n);
	mpz_clear(gcd);
}

#include <unistd.h>
#include <stdlib.h>
#include "config/config.h"
#include "ecm/factor_task.h"
#include "log.h"

int parse_input(char *line, size_t len, run_config config) {
	/* Parse input line */
	int range_s = 0, range_e = 0;
	int sret = 0;

	/* ABORT command */
	sret = sscanf(line, "ABORT %d-%d", &range_s, &range_e);
	if (sret > 0) {
		if (sret == 1) range_e = range_s;
		for (; range_s <= range_e; range_s++) {
			factor_task task = factor_task_by_id_lock(range_s, config);
			if (task != NULL) {
				task->done = true;
				LOG_INFO("[CMD] ABORT id %d", range_s);
				pthread_mutex_unlock(task->mutex);
			} else {
				LOG_WARNING("[CMD] ABORT id %d not found", range_s);
			}
		}
		return 1;
	}

	/* PRIORITY command */
	int prio = 0;
	sret = sscanf(line, "PRIORITY %d %d-%d", &prio, &range_s, &range_e);
	if (sret > 0) {
		if (sret == 1) range_e = range_s;
		for (; range_s < range_e; range_s++) {
			factor_task task = factor_task_by_id_lock(range_s, config);
			if (task != NULL) {
				task->priority = prio;
				LOG_INFO("[CMD] PRIORITY id %d set to %d", range_s, prio);
				pthread_mutex_unlock(task->mutex);
			} else {
				LOG_WARNING("[CMD] PRIORITY id %d not found", range_s);
			}
		}
		return 1;
	}

	/* EFFORT command */
	int effort = 0;
	sret = sscanf(line, "EFFORT %d", &effort);
	if (sret > 0) {
		config->effort_max = effort;
		LOG_INFO("[CMD] EFFORT max_effort set to %d", effort);
		return 1;
	}

	/* INPUT FINISHED command */
	if (strstr(line, "INPUT FINISHED") != NULL) {
		config->input_finished = true;
		LOG_INFO("[CMD] INPUT FINISHED");
		return 1;
	}

	/* IMPLICIT ADD command*/
	int id = 0;
	char *mpnum;
	sret = sscanf(line, "%d %m[0-9]", &id, &mpnum);
	if (sret > 1) {
		mpz_t n;
		mpz_init_set_str(n, mpnum, 10);
		int n_bitwidth = mpz_sizeinbase(n, 2);
		if (n_bitwidth > BITWIDTH) {
			LOG_WARNING("Input line: %d bit number is too large for this build (maximum: %d bits)",
						n_bitwidth,
						BITWIDTH);
		} else if (mpz_cmp_ui(n, 0) < 1) {
			LOG_WARNING("Input line: %d bit number is zero (string: %s bits)", n_bitwidth, line);
		} else {
			factor_task task = factor_task_new(id, n, config);
			factor_task_enqueue(task, config);
		}
		free(mpnum);
		return 1;
	}
	return 0;
}

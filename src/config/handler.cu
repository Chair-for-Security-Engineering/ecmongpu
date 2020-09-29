#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "config/ini.h"
#include "config/handler.h"
#include "ecm/ecm.h"
#include "log.h"

int config_handler(void *user, const char *section, const char *name, const char *value) {

	run_config pconfig = (run_config) user;

#define MATCH(s, n) strcmp(section, s) == 0 && strcmp(name, n) == 0

	if (MATCH("general", "logfile")) {
		if (logfile_open(value) == -1) {
			LOG_FATAL("Error opening logfile %s", value);
			exit(EXIT_FAILURE);
		}
		LOG_INFO("Writing to logfile %s", value);
	} else if (MATCH("general", "loglevel")) {
		loglevel_set(atoi(value));
	} else if (MATCH("general", "mode")) {
		if (strcmp(value, "server") == 0) {
			pconfig->mode = CFG_MODE_SERVER;
    } else if (strcmp(value, "file") == 0) {
			pconfig->mode = CFG_MODE_FILE;
		} else {
			LOG_FATAL("Mode '%s' not supported", value);
		}
	} else if (MATCH("general", "random")) {
		if (strcmp(value, "true") == 0) {
			pconfig->random = true;
			LOG_INFO("Using random seed for RNG.");
		} else if (strcmp(value, "false") == 0) {
			pconfig->random = false;
			LOG_INFO("Using static seed for RNG.");
		}
	} else if (MATCH("cuda", "streams")) {
		pconfig->n_cuda_streams = atoi(value);
	} else if (MATCH("cuda", "use_const_memory")) {
		if (strcmp(value, "false") == 0) {
			pconfig->cuda_use_const_memory = false;
		}
	} else if (MATCH("cuda", "threads_per_block")) {
		if (strcmp(value, "auto") == 0) {
			pconfig->cuda_autotune = true;
		} else {
			pconfig->cuda_autotune = false;
			pconfig->cuda_threads_per_block = atoi(value);
			if (pconfig->cuda_threads_per_block % 2 != 0 || pconfig->cuda_threads_per_block > BATCH_JOB_SIZE) {
				LOG_FATAL("CUDA threads per block has to be even and smaller than %d", BATCH_JOB_SIZE);
				exit(EXIT_FAILURE);
			}
			pconfig->cuda_blocks = BATCH_JOB_SIZE / pconfig->cuda_threads_per_block;
		}
	} else if (MATCH("ecm", "powersmooth")) {
		if (strcmp(value, "false") == 0) {
			pconfig->powersmooth = false;
		}
	} else if (MATCH("ecm", "find_all_factors")) {
		if (strcmp(value, "true") == 0) {
			LOG_INFO("Finding all factors");
			pconfig->ecm_done = &ecm_fully_done;
		}
	} else if (MATCH("ecm", "b1")) {
		pconfig->b1 = atoi(value);
	} else if (MATCH("ecm", "b1chain")) {
		pconfig->b1chain = strdup(value);
	} else if (MATCH("ecm", "stage1.check_all")) {
		if (strcmp(value, "false") == 0) {
			LOG_INFO("[Stage 1] Checking only points off curve");
			pconfig->stage1.check_all = false;
		}
	} else if (MATCH("ecm", "stage2.enabled")) {
		if (strcmp(value, "false") == 0) {
			LOG_INFO("Disabling ECM Stage 2");
			pconfig->stage2.enabled = false;
		}
	} else if (MATCH("ecm", "stage2.window_size")) {
		pconfig->stage2.window_size = atoi(value);
	} else if (MATCH("ecm", "stage2.check_all")) {
		if (strcmp(value, "false") == 0) {
			LOG_INFO("[Stage 2] Checking only points off curve");
			pconfig->stage1.check_all = false;
		}
	} else if (MATCH("ecm", "b2")) {
		pconfig->b2 = atoi(value);
	} else if (MATCH("ecm", "effort")) {
		pconfig->effort_max = atoi(value);
	} else if (MATCH("server", "port")) {
		pconfig->server.port = atoi(value);
	} else if (MATCH("file", "input")) {
		pconfig->files.infile_str = strdup(value);
	} else if (MATCH("file", "output")) {
		pconfig->files.outfile_str = strdup(value);
	} else if (MATCH("ecm", "curve_gen")) {
		pconfig->curve_gen = atoi(value);
		if (pconfig->curve_gen < 0 || pconfig->curve_gen >= job_generators_len) {
			LOG_FATAL("Wrong curve generation selected. Must be 0 to %d.", job_generators_len - 1);
			exit(EXIT_FAILURE);
		} else {
			pconfig->generate_job = job_generators[pconfig->curve_gen];
		}
	} else {
		return 0;  /* unknown section/name, error */
	}
	return 1;
}

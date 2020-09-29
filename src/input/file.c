#include "input/input.h"
#include <unistd.h>
#include <sys/stat.h>
#include "log.h"

static int file_input(char **line, size_t *len, run_config config) {
	int ret;
	struct stat st;
	while (ret = getline(line, len, config->files.infile_fp)) {
		// EOF read
		if (ret == -1) {
			// Is a pipe, wait for further input
			if (!fstat(fileno(config->files.infile_fp), &st) && S_ISFIFO(st.st_mode)) {
				sleep(0.1);
				continue;
			} else {
				return -1;
			}
		} else {
			return 0;
		}
	}
	// never reached
	return 0;
}

void *file_input_thread_run(void *configv) {
	LOG_WARNING("input thread start");
	run_config config = (run_config) configv;
	if (config->files.infile_fp == NULL) {
		LOG_FATAL("Error opening input file");
		exit(EXIT_FAILURE);
	}
	int linecount = 0;
	char *line = NULL;
	size_t len = 0;
	/* Read file line by line */
	while (file_input(&line, &len, config) != -1) {
		linecount++;
		LOG_DEBUG("Input line %i: %s", linecount, line);
		if (!parse_input(line, len, config)) {
			LOG_WARNING("Malformed input line %d: %s", linecount, line);
		}
	}

	config->input_finished = true;
	return NULL;
}



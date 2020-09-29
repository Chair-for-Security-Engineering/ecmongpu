#include "config/config.h"
#include "ecm/factor_task.h"
#include "ecm/ecm.h"
#include "ecc/twisted_edwards.h"
#include "cudautil.h"
#include "log.h"
#include "config/handler.h"
#include <sys/stat.h>
#include <time.h>

#include <unistd.h>
#include <sys/socket.h>
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>


pthread_mutex_t mutex_factor_task = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex_output = PTHREAD_MUTEX_INITIALIZER;

void factor_task_push(factor_task_heap_t *h, int priority, factor_task data) {
	pthread_mutex_lock(&mutex_factor_task);

	if (h->len + 1 >= h->size) {
		h->size = h->size ? h->size * 2 : 4;
		h->nodes = (factor_task_node_t *) realloc(h->nodes, h->size * sizeof(factor_task_node_t));
	}
	int i = h->len + 1;
	int j = i / 2;
	while (i > 1 && h->nodes[j].priority > priority) {
		h->nodes[i] = h->nodes[j];
		i = j;
		j = j / 2;
	}
	h->nodes[i].priority = priority;
	h->nodes[i].data = data;
	h->len++;

	pthread_mutex_unlock(&mutex_factor_task);
}

factor_task factor_task_pop(factor_task_heap_t *h) {
	return factor_task_pop_priority(h, NULL);
}


factor_task factor_task_pop_priority(factor_task_heap_t *h, int *priority) {
	pthread_mutex_lock(&mutex_factor_task);
	int i, j, k;
	if (!h->len) {
		pthread_mutex_unlock(&mutex_factor_task);
		return NULL;
	}
	factor_task data = h->nodes[1].data;

	if (priority != NULL) {
		(*priority) = h->nodes[1].priority;
	}

	h->nodes[1] = h->nodes[h->len];

	h->len--;

	i = 1;
	while (i != h->len + 1) {
		k = h->len + 1;
		j = 2 * i;
		if (j <= h->len && h->nodes[j].priority < h->nodes[k].priority) {
			k = j;
		}
		if (j + 1 <= h->len && h->nodes[j + 1].priority < h->nodes[k].priority) {
			k = j + 1;
		}
		h->nodes[i] = h->nodes[k];
		i = k;
	}

	pthread_mutex_unlock(&mutex_factor_task);
	return data;
}


int file_output(char* string, run_config config){
  if(config->files.outfile_fp != NULL){
	  pthread_mutex_lock(&mutex_output);
		fputs(string, config->files.outfile_fp);
    fflush(config->files.outfile_fp);
		pthread_mutex_unlock(&mutex_output);
  }
  return 0;
}


int socket_output(char* string, run_config config){
  if(config->server.client_socket != -1){
	  pthread_mutex_lock(&mutex_output);
		send(config->server.client_socket, string, strlen(string), 0);
    fflush(config->files.outfile_fp);
		pthread_mutex_unlock(&mutex_output);
  }
  return 0;
}


void run_config_set_defaults(run_config c) {
	c->b1 = 1000;
	c->b1chain = NULL;

	c->powersmooth = true;
	c->b2 = 10000;

	c->mode = CFG_MODE_FILE;

	c->server.port = -1;
	c->server.socket = -1;
	c->server.client_socket = -1;

	c->effort_max = 10;
	c->n_cuda_streams = 2;
	c->curve_gen = 2;
	c->generate_job = job_generators[c->curve_gen];

	c->cuda_use_const_memory = true;

	c->stage1.check_all = true;

	c->stage2.window_size = (2 * 3 * 5 * 7 * 11);

	c->stage2.enabled = true;
	c->stage2.check_all = true;

	c->cuda_autotune = false;

	c->cuda_threads_per_block = BLOCK_SIZE;
	c->cuda_blocks = BATCH_JOB_SIZE / BLOCK_SIZE;
	c->ecm_done = &ecm_factor_found_done;

	c->output = &socket_output;
	c->output = &file_output;

	c->files.outfile_fp = NULL;
	c->files.infile_fp = NULL;

	c->input_finished = false;

	c->task_tree_root = NULL;

	c->random = true;
}


void run_config_init(run_config c, gmp_randstate_t gmprand) {

	c->stage1.host_bound_naf = NULL;
	c->stage1.bound_naf_digits = -1;

	c->rand = gmprand;

	c->cuda_streams = (cudaStream_t *) malloc(sizeof(cudaStream_t) * c->n_cuda_streams * c->devices);
	c->host_threads = (pthread_t *) malloc(sizeof(pthread_t) * c->n_cuda_streams * c->devices);

	/* Create context for each device */
	c->dev_ctx = (dev_ctx *) malloc((unsigned int) (c->devices * sizeof(dev_ctx)));
	for (int dev = 0; dev < c->devices; dev++) {
		c->dev_ctx[dev].stage1.dev_bound_naf = NULL;

		int offset = c->n_cuda_streams * dev;
		cudaSetDevice(dev);
		for (int stream = 0; stream < c->n_cuda_streams; stream++) {
			cudaStreamCreateWithFlags(&c->cuda_streams[offset + stream], cudaStreamNonBlocking);
		}
	}


	c->factor_tasks_queue = (factor_task_heap_t *) calloc(1, sizeof(factor_task_heap_t));

	c->task_tree_root = NULL;

  if(c->mode == CFG_MODE_SERVER) {
	  c->output = &socket_output;
  } else if(c->mode == CFG_MODE_FILE){
	  c->output = &file_output;
  } else {
    LOG_FATAL("No corresponding output function for mode available.");
  }
}

/* Parse a single option. */
static error_t parse_opt(int key, char *arg, struct argp_state *state) {
	run_config config = (run_config) state->input;

	switch (key) {
		case 1:
			if(arg) config->b1 = atoi(arg);
			break;
		case 2:
			if(arg) config->b2 = atoi(arg);
			break;
		case 3:
			if(arg) config->b1chain = strdup(arg);
			break;
		case 'e':
			if(arg) config->effort_max = atoi(arg);
			break;
		case 'p':
			if(arg) config->server.port = atoi(arg);
			break;
		case 'c':
			if(arg) config->files.cfgfile_str = strdup(arg);
			break;
		case 'l':
			config->mode = CFG_MODE_SERVER;
			break;
		case 'f':
			config->mode = CFG_MODE_FILE;
			break;
		case 4:
			log_set_color(true);
			break;
		case 5:
			if(arg) loglevel_set(atoi(arg));
			break;
		case 's':
			loglevel_set(LOG_LEVEL_FATAL);
			break;
		case ARGP_KEY_ARG:
		case ARGP_KEY_END:
			/* No regular arguments */
			if (state->arg_num != 0)
				argp_usage(state);
			break;
		default:
			return ARGP_ERR_UNKNOWN;
	}
	return 0;
}

/* The options we understand. */
static struct argp_option options[] = {
		{"b1",     	1,   "int",  0, "First stage bound"},
		{"b2",     	2,   "int",  0, "Second stage bound"},
		{"b1chain",	3,   "string",  0, "Filename of chain for first stage bound"},
		{"listen", 	'l', 0,      0, "Listen on port"},
		{"file",   	'f', 0,      0, "Input files"},
		{"port",   	'p', "port", 0, "Port for server mode"},
		{"config", 	'c', "FILE", 0, "Config file"},
		{"effort", 	'e', "int",  0, "Effort per input number"},
		{"use-color", 	 4,   0,  0, "Enable color output"},
		{"log",          5,   "int",  0, "Set log level"},
		{"silent",      's',   0,  0, "Set log level"},
		{0}
};

static struct argp argp = {
		options,
		parse_opt,
		"",
		"co-ecm -- factor numbers with ECM on NVIDIA GPUs"
};

void run_config_read(run_config config, int argc, char** argv){
	run_config_set_defaults(config);
	/* Parse command line parameters for config file */
	argp_parse(&argp, argc, argv, 0, 0, config);


	if (config->files.cfgfile_str && ini_parse(config->files.cfgfile_str, config_handler, config) < 0) {
		LOG_INFO("Can't load config %s", config->files.cfgfile_str);
		exit(EXIT_FAILURE);
	}

	/* Parse again to overwrite with cli */
	argp_parse(&argp, argc, argv, 0, 0, config);

	// Check configuration essentials
	if (config->mode == CFG_MODE_FILE) {
	  config->files.outfile_fp = fopen(config->files.outfile_str, "a");
    	  if (config->files.outfile_fp == NULL) {
    	    LOG_FATAL("Could not open output file %s", config->files.outfile_str);
    	    exit(EXIT_FAILURE);
    	  }

    	  config->files.infile_fp = fopen(config->files.infile_str, "r");
    	  if (config->files.infile_fp == NULL) {
    	    LOG_FATAL("Could not open input file %s", config->files.infile_str);
    	    exit(EXIT_FAILURE);
    	  }

	}
	if (config->mode == CFG_MODE_SERVER) {
		if (config->server.port == -1) {
			LOG_FATAL("No server port given.");
			exit(-1);
		}
	}

	run_config_log(config);
}


void run_config_free(run_config c) {
	/* Cleanup task */
	for (int stream = 0; stream < c->n_cuda_streams * c->devices; stream++) {
		CUDA_SAFE_CALL_NO_SYNC(cudaStreamSynchronize(c->cuda_streams[stream]));
		/* Free batch memory */
		if (c->stage1.host_bound_naf != NULL) {
			cudaFreeHost(c->stage1.host_bound_naf);
		} else {
			LOG_WARNING("Cleaning up uninitialized task.");
		}
		cudaStreamDestroy(c->cuda_streams[stream]);
	}

	if (c->stage1.host_bound_naf != NULL) {
		free(c->stage1.host_bound_naf);
	} else {
		LOG_WARNING("Cleaning up uninitialized task.");
	}
}

void run_config_log(run_config c) {
	LOG_INFO("[Config]");
	LOG_INFO("\tinput: %s", c->files.infile_str);
	LOG_INFO("\toutput: %s", c->files.outfile_str);
	LOG_INFO("[Server]");
	LOG_INFO("\tport: %d", c->server.port);

	LOG_INFO("  [ECM]");
	LOG_INFO("\teffort_max: %d", c->effort_max);
	LOG_INFO("\t[Stage 1]");
	LOG_INFO("\tb1: %d (%ssmoothness)", c->b1, c->powersmooth ? "power" : "");
	LOG_INFO("\t%s", c->stage1.check_all ? "Checking all points for factors" : "Checking only non-curve points for factors");

	LOG_INFO("\t[Stage 2]");
	LOG_INFO("\t%s", c->stage2.enabled ? "enabled" : "disabled");
	LOG_INFO("\tb2: %d", c->b2);
	LOG_INFO("\twindow size: %d", c->stage2.window_size);
	LOG_INFO("\t%s", c->stage2.check_all ? "Checking all points for factors" : "Checking only non-curve points for factors");

	LOG_INFO("  [CUDA]");
	LOG_INFO("\tn_cuda_streams: %d", c->n_cuda_streams);
	LOG_INFO("\tcurve_gen: %s", job_generators_names[c->curve_gen]);
	LOG_INFO("\tcuda_threads_per_block: %d", c->cuda_threads_per_block);
	LOG_INFO("\tcuda_blocks: %d", c->cuda_blocks);
}


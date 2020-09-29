#include <stdlib.h>
#include <time.h>
#include <gmp.h>
#include <search.h>
#include <errno.h>
#include <cuda_runtime.h>
#include "cudautil.h"

#include "input/input.h"
#include "mp/mp.h"
#include "mp/mp_montgomery.h"
#include "mp/gmp_conversion.h"
#include "ecc/twisted_edwards.h"
#include "ecm/batch.h"
#include "ecm/factor_task.h"
#include "ecm/ecm.h"
#include "ecc/naf.h"
#include "version.h"
#include "ecm/stage1.h"
#include "ecm/stage2.h"

#include <signal.h>
#include <unistd.h>

#include "log.h"


void print_usage() {
	printf("Usage: ecm -c config\n");
	printf("\t-c \tconfiguration file\n");
}

static bool run = true;


void siginthandler(int p) {
	LOG_FATAL("SIGINT received, aborting");
	run = false;
	exit(1);
}

struct thread_args {
	run_config config;
	batch_naf *batch;
	size_t stream;
	int jobs;
};

void *thread_run(void *argsv) {
	struct thread_args *args = (struct thread_args *) argsv;
	struct timespec start = {0, 0}, end = {0, 0};
	double tmp_time, c_per_sec;

	while (run) {
		/* Wait if input is a pipe */
		if (args->config->factor_tasks_queue->len <= 0) {
			if (args->config->input_finished) {
				break;
			} else {
				sleep(0.5);
				continue;
			}
		}

		/* Set this threads device */
		cudaSetDevice(args->batch->host[args->stream]->device);

		// Collect tasks	
		args->batch->host[args->stream]->n_jobs = 0;
		for (int job = 0; job < BATCH_JOB_SIZE; job++) {
			args->batch->host[args->stream]->tasks_id[job] = 0;
		}
		for (int job = 0; job < BATCH_JOB_SIZE; job++) {
			factor_task task = factor_task_get_next(args->config);
	
			/* Either queues are empty or a good task was retrieved */
			if (task) {
				args->batch->host[args->stream]->n_jobs++;
				compute_batch_job(task, args->config, args->batch->host[args->stream], job);
				/* Set batch id for this job to correct task */
				args->batch->host[args->stream]->tasks_id[job] = task->id;
			} else {
				break;
			}
		}

		if (args->batch->host[args->stream]->n_jobs == 0) {
			LOG_WARNING("[Thread %d] No tasks left for stage 1", args->stream);
			continue;
		}

		LOG_INFO("[Thread %d] %d tasks in batch for Stage 1", args->stream, args->batch->host[args->stream]->n_jobs);

		/* Stage 1 */
		clock_gettime(CLOCK_MONOTONIC, &start);
		ecm_stage1(args->config, args->batch, args->stream);
		clock_gettime(CLOCK_MONOTONIC, &end);
		tmp_time = (((double) end.tv_sec + 1.0e-9 * end.tv_nsec) - ((double) start.tv_sec + 1.0e-9 * start.tv_nsec)) *
				   1000;
		c_per_sec = ((double) (args->batch->host[args->stream]->n_jobs)) / (tmp_time / 1000);

		LOG_INFO("[Device %i] [Thread %i] Stage 1 Performance: %d curves in %.0fms (%.0f c/s)",
				 args->batch->host[args->stream]->device,
				 args->stream,
				 args->batch->host[args->stream]->n_jobs,
				 tmp_time,
				 c_per_sec);

		if (run && args->config->stage2.enabled) {
			clock_gettime(CLOCK_MONOTONIC, &start);
			ecm_stage2(args->config, args->batch, args->stream);
			clock_gettime(CLOCK_MONOTONIC, &end);
			tmp_time =
					(((double) end.tv_sec + 1.0e-9 * end.tv_nsec) - ((double) start.tv_sec + 1.0e-9 * start.tv_nsec)) *
					1000.0;

			c_per_sec = ((double) (args->batch->host[args->stream]->n_jobs)) / (tmp_time / 1000);
			LOG_INFO("[Device %i] [Thread %i] Stage 2 Performance: %d curves in %.0fms (%.0f c/s)",
					 args->batch->host[args->stream]->device,
					 args->stream,
					 args->batch->host[args->stream]->n_jobs,
					 tmp_time,
					 c_per_sec);
		}
		args->jobs += args->batch->host[args->stream]->n_jobs;

		if (args->stream == 0) {
			LOG_INFO("Task Queue: %i tasks", args->config->factor_tasks_queue->len);
		}
	}
	return NULL;
}

void ecm_init(batch_naf *batch, run_config config) {
	if (config->cuda_autotune) {
		int numBlocks, numBlocksMax = 0;
		for (int threads = 32; threads <= 1024; threads += 32) {
			CUDA_SAFE_CALL(
					cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, &cuda_tw_ed_smul_naf_batch,
																  threads, sizeof(mon_info) * threads)
			);
			if (numBlocks > numBlocksMax) {
				numBlocksMax = numBlocks;
				config->cuda_threads_per_block = threads;
				LOG_WARNING("CUDA Determined number of blocks for %d threads:%d", threads, numBlocks);
			}
		}
		config->cuda_blocks = BATCH_JOB_SIZE / config->cuda_threads_per_block;
	}

	LOG_INFO("CUDA using %d threads per block.", config->cuda_threads_per_block);
	LOG_INFO("CUDA using %d blocks.", config->cuda_blocks);

	batch_allocate(config, batch);
	LOG_INFO("Initial Task Queue: %i tasks", config->factor_tasks_queue->len);

	/* init */
	ecm_stage1_init(config);

	if (config->stage2.enabled) {
		ecm_stage2_init(config);
		ecm_stage2_initbatch(config, batch);
	}
}

void ecm(batch_naf *batch, run_config config) {

	int jobs = 0;

	struct timespec full_start = {0, 0}, full_end = {0, 0};
	clock_gettime(CLOCK_MONOTONIC, &full_start);

	struct thread_args *targs = (struct thread_args *) malloc(
			config->devices * config->n_cuda_streams * sizeof(struct thread_args));

	for (int stream = 0; stream < config->devices * config->n_cuda_streams; stream++) {
		targs[stream].config = config;
		targs[stream].batch = batch;
		targs[stream].stream = stream;
		targs[stream].jobs = 0;
		pthread_create(&config->host_threads[stream], NULL, &thread_run, (void *) &targs[stream]);
	}
	for (int stream = 0; stream < config->devices * config->n_cuda_streams; stream++) {
		pthread_join(config->host_threads[stream], NULL);
		jobs += targs[stream].jobs;
	}
	config->output("DONE\n", config);


	clock_gettime(CLOCK_MONOTONIC, &full_end);
	double full_time = (((double) full_end.tv_sec + 1.0e-9 * full_end.tv_nsec) -
						((double) full_start.tv_sec + 1.0e-9 * full_start.tv_nsec)) * 1000;
	float c_per_sec = ((float) (jobs)) / (full_time / 1000);
	LOG_INFO("[Total] Stage 1%s Performance: %d curves in %.0fms (%.0f c/s)", (config->stage2.enabled ? "&2" : ""),
			 jobs, full_time, c_per_sec);


	LOG_INFO("Final Task Queue: %i tasks", config->factor_tasks_queue->len);

}




int main(int argc, char *argv[]) {

	struct sigaction a;
	a.sa_handler = siginthandler;
	a.sa_flags = 0;
	sigemptyset( &a.sa_mask );
	sigaction(SIGINT, &a, NULL);

	signal(SIGPIPE, SIG_IGN);

	cudaDeviceReset();
	cudaSetDeviceFlags(cudaDeviceBlockingSync);


	LOG_INFO("co-ecm version: git-%s-%s", GIT_BRANCH, GIT_COMMIT_HASH);
	LOG_INFO("\tBuild type: %s", BUILD_TYPE);

	LOG_INFO("\tBitwidth: %i", BITWIDTH);
	LOG_INFO("\tLimbs: %i", LIMBS);


	/* Initialize config, read config file and populate config struct */
	run_config config = (struct _run_config *) malloc(sizeof(struct _run_config));
	run_config_read(config, argc, argv);

	// set config run flag
	config->run = &run;

	CUDA_SAFE_CALL(cudaGetDeviceCount(&config->devices));
	if (config->devices == 0) {
		LOG_FATAL("No CUDA device available.");
		exit(EXIT_FAILURE);
	}

	/* Iterate and log available CUDA devices */
	for (int d = 0; d < config->devices; d++) {
		struct cudaDeviceProp cuda_properties;
		CUDA_SAFE_CALL(cudaGetDeviceProperties(&cuda_properties, d));
		LOG_INFO("CUDA Device %i", d);
		LOG_INFO("\tName:\t%s", cuda_properties.name);
		LOG_INFO("\tGlobal Memory:\t%zu bytes", cuda_properties.totalGlobalMem);
		LOG_INFO("\tConstant Memory:\t%zu bytes", cuda_properties.totalConstMem);
		LOG_INFO("\tShared Mem per Block:\t%zu bytes", cuda_properties.sharedMemPerBlock);
		LOG_INFO("\t32bit Registers per Block:\t%d", cuda_properties.regsPerBlock);
		LOG_INFO("\tMax Threads per Block:\t%d", cuda_properties.maxThreadsPerBlock);
		LOG_INFO("\tWarpsize:\t%d threads", cuda_properties.warpSize);
		LOG_INFO("\tMultiprocessors:\t%d", cuda_properties.multiProcessorCount);
	}


	/* Initialize and (statically seed) RNG */
	gmp_randstate_t gmprand;
	gmp_randinit_default(gmprand);
	if (config->random) {
		srand(time(NULL));
		gmp_randseed_ui(gmprand, rand());
	} else {
		gmp_randseed_ui(gmprand, 0);
	}
	run_config_init(config, gmprand);

	for (int d = 0; d < config->devices; d++) {
		LOG_INFO("== Using device %d ==", d);
	}

	LOG_INFO("CUDA Configuration");
	LOG_INFO("\tConcurrent Streams: %d", config->n_cuda_streams);
	LOG_INFO("\tCurves per Batch:   %d", BATCH_JOB_SIZE);
	LOG_INFO("\tThreads per Block:  %d", config->cuda_threads_per_block);
	LOG_INFO("\tBlocks per Stream:  %d", config->cuda_blocks);

	LOG_INFO("ECM Configuration");
	LOG_INFO("\tB1 (%ssmooth):     %d", (config->powersmooth ? "power" : ""), config->b1);
	LOG_INFO("\tB2:              %d", config->b2);
	LOG_INFO("\tMax Effort:      %d", config->effort_max);
	LOG_INFO("\tCurve generator: %s", job_generators_names[config->curve_gen]);



	pthread_t input_t;
	if (config->mode == CFG_MODE_FILE) {
		pthread_create(&input_t, NULL, &file_input_thread_run, (void *) config);
	} else if (config->mode == CFG_MODE_SERVER) {
		pthread_create(&input_t, NULL, &server_thread_run, (void *) config);
	} else {
		LOG_FATAL("No correct input selected.");
		exit(EXIT_FAILURE);
	}

	batch_naf batch;
	ecm_init(&batch, config);

	while(run){
		ecm(&batch, config);
		if (config->input_finished && config->mode == CFG_MODE_FILE) break;
		config->input_finished = false;
	}

	/* Cleanup */
	logfile_close();
	gmp_randclear(gmprand);

	return 0;
}


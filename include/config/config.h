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

#ifndef CO_ECM_CONFIG_H
#define CO_ECM_CONFIG_H

typedef struct _run_config *run_config;

#include <gmp.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pthread.h>

#include "ecc/naf.h"
#include "ecc/twisted_edwards.h"
#include "ecm/batch.h"
#include "ecm/factor_task.h"

#define CFG_MODE_SERVER 1
#define CFG_MODE_FILE 2

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*ecm_done_fn)(factor_task task);

typedef int (*output_fn)(char *string, run_config config);

typedef struct {
	int priority;
	factor_task data;
} factor_task_node_t;

typedef struct {
	factor_task_node_t *nodes;
	int len;
	int size;
} factor_task_heap_t;


typedef struct _stage2_global {
	struct {
		size_t naf_size;
		naf_t naf;
	} w;

	struct {
		size_t n;            /**< number of babysteps */
		size_t *naf_size;    /**< number of naf_digits in each babystep_naf */
		naf_t *naf;            /**< naf scalars for all babysteps */
	} babysteps;

	size_t giantsteps_n;

	mp_limb *is_prime;        /**< marks wheter either of (v*w +- u) is prime */
} stage2_global;


typedef struct __dev_ctx {
	struct {
		naf_t dev_bound_naf;    /**< Scalar (= primorial(b1)) in NAF form on CUDA device */
	} stage1;

	struct {
		stage2_global *global_dev;
	} stage2;
} dev_ctx;

typedef struct _run_config {

	int b1;                    /**< Smoothness bound for ECM stage 1 */
	char *b1chain;                    /**< Filename of ECM stage 1 bound chain file */

	bool powersmooth;        /**< Compute stage 1 scalar as lcm(2, ..., b1) when true, as primorial(b1) if false; */

	int b2;                    /**< Smoothness bound for ECM stage 2 */

	int effort_max;            /**< How many effort to spend on factoring this number. */

	output_fn output;
	bool input_finished;

	int mode;

	struct {
		char *infile_str;        /**< Path to input file */
		FILE *infile_fp;        /**< Handle for input file */

		char *outfile_str;        /**< Path to output file */
		FILE *outfile_fp;        /**< Handle for output file */

		char *cfgfile_str;        /**< Path to output file */
		FILE *cfgfile_fp;        /**< Handle for output file */
	} files;

	struct {
		int port;
		int socket;
		int client_socket;
	} server;


	bool random;            /**< If false, statically seed RNG, else use random seed */

	/**
	 * Stage one parameters.
	 */
	struct {
		naf_t host_bound_naf;    /**< Scalar (= primorial(b1)) in NAF form on host */
		int bound_naf_digits;    /**< Number of digits in NAF form scalar */
		bool check_all;            /**< If true, check all points for factors, if falso, check only those off curve */
	} stage1;

	struct {
		bool enabled;            /**< If true, perform stage 2 */
		bool check_all;            /**< If true, check all points for factors, if falso, check only those off curve */
		int window_size;

		stage2_global *global_host;
	} stage2;


	int devices;
	dev_ctx *dev_ctx;


	__gmp_randstate_struct *rand;    /**< GMP RNG struct to be used with this factor task, e.g. for curve generation. */
	int n_cuda_streams;                /**< Number of CUDA streams to use for this task */
	cudaStream_t *cuda_streams;        /**< Array containing the (reused) CUDA streams for this task (to avoid stream creation overhead */
	pthread_t *host_threads;
	int cuda_autotune;        /**< Enable or disable autotuning block/thread config */
	int cuda_threads_per_block;        /**< Number of threads executed within one block for this task */
	int cuda_blocks;                /**< Number of blocks (each with cuda_threads_per_block threads) to execute */
	bool cuda_use_const_memory;        /**< If true, use constant memory for scalar in stage 1 (if enough const memory available */

	int curve_gen;
	job_generator generate_job;        /**< Function pointer for curve generation algorithm for this task */

	factor_task_heap_t *factor_tasks_queue;    /**< Heap storing enqueued tasks */

	ecm_done_fn ecm_done;

	void *task_tree_root;

	bool *run;

} *run_config;


/**
 * Add a factor task to a (priority) heap with priority.
 *
 * Lower value of parameter priority means higher priority
 * @param h Heap to insert into
 * @param priority Priority to insert with (lower value -> higher priority)
 * @param data	factor_task to insert
 */
void factor_task_push(factor_task_heap_t *h, int priority, factor_task data);

/**
 * Pop one factor task of the (priority) heap
 *
 * @param h Heap to retrieve factor_task from
 * @return factor_task with highest priority
 */
factor_task factor_task_pop(factor_task_heap_t *h);


/**
 * Pop one factor task of the (priority) heap and return its previous priority in this heap.
 *
 * @param h Heap to retrieve factor_task from
 * @param priority Return parameter that will be set to the popped factor_tasks priority in this heap
 * @return factor_task with highest priority
 *
 */
factor_task factor_task_pop_priority(factor_task_heap_t *h, int *priority);


/**
 * Log a run_config with loglevel INFO.
 *
 * @param c config to log
 */
void run_config_log(run_config c);


/**
 * Free a config struct, free all memory.
 * @param c  config to destroy
 */
void run_config_free(run_config c);


void run_config_read(run_config config, int argc, char **argv);


void run_config_init(run_config c, gmp_randstate_t gmprand);

#ifdef __cplusplus
}
#endif

#endif //CO_ECM_CONFIG_H

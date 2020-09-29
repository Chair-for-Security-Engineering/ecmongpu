#include "ecm/batch.h"
#include "mp/mp.h"
#include "ecc/twisted_edwards.h"
#include "log.h"

__host__
void print_batch_naf(batch_job_naf *batch) {
	for (int job = 0; job < BATCH_JOB_SIZE; job++) {
		printf("Job: %i\n", job);
		curve_tw_ed curve;
		point_tw_ed p;
		mp_copy_cs(curve.d, batch->job.curve_strided.d, job);
		mp_copy_cs(curve.k, batch->job.curve_strided.k, job);
		tw_ed_print_curve(&curve);
		tw_ed_copy_point_cs(&p, &batch->job.point_strided, job);


		mon_info info;
		mp_copy_cs(info.n, batch->job.mon_info_strided.n, job);
		mp_copy_cs(info.R2, batch->job.mon_info_strided.R2, job);
		info.mu = batch->job.mon_info_strided.mu[job];

		tw_ed_print_point_strided(&batch->job.point_strided, job, &info);
		for (int i = 1; i <= NAF_MAX_PRECOMPUTED; i += 2) {
			printf("%d * P:\n", i);
			tw_ed_print_point_strided(&batch->job.precomputed_strided[__naf_to_index(i)], job, &info);
		}
	}
}

__host__
void compute_batch_job(factor_task task, run_config config, batch_job_naf *batch, size_t job) {
	mp_t n;
	mpz_to_mp(n, task->composite);
	mon_info info;
	mon_info_populate(n, &info);

	/* Copy mon info to strided version */
	mp_copy_sc(batch->job.mon_info_strided.n, job, info.n);
	mp_copy_sc(batch->job.mon_info_strided.R2, job, info.R2);
	batch->job.mon_info_strided.mu[job] = info.mu;

	curve_tw_ed curve;
	point_tw_ed p;
	config->generate_job(&p, &curve, &info, config->rand, (void *)&task->p_gkl2016);

	tw_ed_copy_point_sc(&batch->job.point_strided, job, &p);

	/* Copy curve info to strided version */
	mp_copy_sc(batch->job.curve_strided.d, job, curve.d);
	mp_copy_sc(batch->job.curve_strided.k, job, curve.k);

}


__host__
void batch_allocate(run_config config, batch_naf *batch) {
	LOG_DEBUG("Allocating batch memory for %i streams", config->n_cuda_streams * config->devices);

	/* Generate batch pointers */
	batch->host = (batch_job_naf **) malloc(config->devices * config->n_cuda_streams * sizeof(batch_job_naf *));
	batch->dev = (batch_job_naf **) malloc(config->devices * config->n_cuda_streams * sizeof(batch_job_naf *));


	/* Allocate batch memory */
	for (int dev = 0; dev < config->devices; dev++) {
		cudaSetDevice(dev);
		int offset = config->n_cuda_streams * dev;
		for (int stream = 0; stream < config->n_cuda_streams; stream++) {
			CUDA_SAFE_CALL(cudaMalloc(&batch->dev[offset + stream], sizeof(batch_job_naf)));
			CUDA_SAFE_CALL(cudaMallocHost(&batch->host[offset + stream], sizeof(batch_job_naf)));
			batch->host[offset + stream]->config = config;
			batch->host[offset + stream]->device = dev;
		}
	}

}

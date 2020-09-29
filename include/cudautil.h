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

#ifndef CUDA_UTIL_H
#define CUDA_UTIL_H

#include <cuda_runtime.h>
#include <cudautil.h>

/**
 * Macro to wrap calls to CUDA Runtime library and kernels and do error checking.
 * Does not synchronize the device, might return old errors.
 */
#  define CUDA_SAFE_CALL_NO_SYNC(call) do {                                 \
    cudaError_t err = call;                                                    \
    if( cudaSuccess != err) {                                                \
        LOG_FATAL("Cuda Error: %s", cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

/**
 * Macro to wrap calls to CUDA Runtime library and kernels and do error checking.
 * Does synchronize the device after the call to the function.
 */
#  define CUDA_SAFE_CALL(call) do {                                         \
    CUDA_SAFE_CALL_NO_SYNC(call);                                            \
    cudaError_t err = cudaDeviceSynchronize();                                 \
    if( cudaSuccess != err) {                                                \
        LOG_FATAL("Cuda Error: %s", cudaGetErrorString( err) );              \
        exit(EXIT_FAILURE);                                                  \
    } } while (0)

#endif /* CUDA_UTIL_H */

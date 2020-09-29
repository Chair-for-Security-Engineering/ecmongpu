# co-ecm

This software uses one or mutliple Nvidia GPU to factor numbers via Lenstra's
Elliptic Curve Method (ECM).

It operates on "a = -1" twisted Edwards curves with extended projective
coordinates, uses w-NAF point multiplication during stage one of the ECM
algorithm and a baby-step/giant-step approach during stage two. For the first
stage the software also supports custom precomputed addition chains. See the
paper below for more information.

All arithmetic is achieved with a custom fixed bitlength multi-precision
arithmetic implementation. All GPU multi-precision operations are performed in
Montgomery arithmetic.

Stage one and two of the ECM are executed entirely (with exception of GCD
computation) on the GPU, including precomputation (for NAF based scalar
multiplication).


# Paper 
More details about the software and evaluation results are published in 

> *Jonas Wloka, Jan Richter-Brockmann, Colin Stahlke, Thorsten Kleinjung,
> Christine Priplata, Tim Güneysu*:
> **Revisiting ECM on GPUs**.
> 19th International Conference on Cryptology and Network Security (CANS 2020),
> December 14-16, 2020, 


## Building

### Dependencies

Install the following dependencies before compiling this software

 - For most host-side computations, the [GNU Multi Precision Arithmetic
   Library](https://gmplib.org/) is used. 
 - [OpenMP](https://www.openmp.org/) is (scarcely as of now) used to
   parallelize some jobs on the CPU.
 - As this software targets CUDA capable GPUs, CUDA itself needs to be
   installed, in particular the Nvidia compiler `nvcc` and the [CUDA library in
   Version 10](https://developer.nvidia.com/cuda-downloads).
 - To run the included benchmarks, `python` is needed with at least version 3


### Compilation

The build system CMake is used. Install CMake for your system. If you are on a
linux-based system, you will also need GNU make to actually build the software,
as well as a current C compiler.

See the top of the `CMakeLists.txt` for build time configuration options.

 * `BITWIDTH` gives the maximum bits any input number can have. The higher this
   value, the slower all arithmetic runs.
 * The option `BATCH_JOB_SIZE` determines the number of curves in one batch
   processing.


To compile the software, create a build folder and change into it 

```
$ mkdir build 
$ cd build 
``` 

execute CMake with the path to this source directory to generate Makefiles 

``` 
$ cmake /path/to/source 
```
Finally call `make` to build
the binaries:

```
$ make cuda-ecm
```

### Testing to be sure all multi precision and elliptic curve arithmetic works
correctly on your system, make can generate and execute a test suite after
CMake setup via 

```
$ make all test
 ```

### Running integrated benchmarks
Two sample benchmarks can be run with

```
$ make bench
```



## Usage

```
Usage: ecm -c config.ini 

    -c 	configuration file
 ```

### Configuration file

An example configuration file is provided in the
`example` directory as `config.ini`.  All configuration options, as well as
their default values are described in this file.



## Example usage 

The following snippet builds the `cuda-ecm` binary and tries
to factor a small number of provided 192-bit numbers with a 48-bit and 144-bit
factor in file `example/input.txt`. For a more challenging approach change the 
input file to `example/input2.txt`, a file with 32768 numbers of 100-bit and
91-bit factors.

```

$ cd /path/to/source 
$ mkdir build 
$ cd build 
$ cmake ..
$ make cuda-ecm
$ ./bin/cuda-ecm -c ../example/config.ini

```

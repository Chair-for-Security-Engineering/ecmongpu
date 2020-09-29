cmake_minimum_required(VERSION 0.7)
project(co-ecm C CUDA)

set(VERSION "v0.8")
set(CMAKE_BUILD_TYPE Release)

set(BUILD_BENCHMARKS 1)


# Options and Settings

# Log level: VERBOSE, DEBUG, INFO, WARNING, ERROR, FATAL
set(LOG_LEVEL INFO)

# Multi precision limbs, use 32 or 64 bit datatype
set(LIMB_BITS 32)

# Bits of basic multi precision datatype. Has to be 32 bit more than the modulus to be factored
if(DEFINED ENV{BITWIDTH})
	set(BITWIDTH $ENV{BITWIDTH})
else()
	set(BITWIDTH 192)
endif()
message("Building for ${BITWIDTH}-bit moduli")

# Default CUDA threads per block.
set(BLOCK_SIZE 128)

# Curves per single batch.
set(BATCH_JOB_SIZE 32768)

# Window size for w-NAF.
if(DEFINED ENV{WINDOW_SIZE})
  set(NAF_WINDOW_SIZE $ENV{WINDOW_SIZE})
else()
  set(NAF_WINDOW_SIZE 4)
endif()
message("Building with window size w=${NAF_WINDOW_SIZE}")

# Default allocated number of NAF digits for ECM stage 2.
# If needed for large Stage 2 bounds. You will be asked to increase this value and recompile.
set(NAF_STAGE2_DEFAULT_DIGITS 20)

# Use optimized precomputation
if(NOT DEFINED ENV{DISABLE_OPTIMIZED_PRECOMP})
  set(OPTIMIZE_PRECOMP 1)
endif()
if(NOT DEFINED OPTIMIZE_PRECOMP)
  message("Optimized point representation disabled")
endif()

# Choose a Montgomery product algorithm
if(DEFINED ENV{MON_PROD})
  if("$ENV{MON_PROD}" STREQUAL "CIOS")
    set(MON_PROD_CIOS 1)
  endif()
  if("$ENV{MON_PROD}" STREQUAL "CIOS_XMAD")
    set(MON_PROD_CIOS_XMAD 1)
  endif()
  if("$ENV{MON_PROD}" STREQUAL "FIPS")
    set(MON_PROD_FIPS 1)
  endif()
  if("$ENV{MON_PROD}" STREQUAL "FIOS")
    set(MON_PROD_FIOS 1)
  endif()
else ()
  set(MON_PROD_FIPS 1)
endif()


set(COORDINATES_EXTENDED 1)
#set(COORDINATES_INVERTED 1)

# Set the maximum number of registers during compilation. 
# Low values result in excessive spilling to (slow local, ie global ) memory.
#set(GPU_MAX_REG 64)

# Set CUDA architectures to generate binary code for

if(DEFINED ENV{GPU_ARCH})
    set(GPU_ARCHITECTURE $ENV{GPU_ARCH})
    message("Building for CUDA architecture ${GPU_ARCHITECTURE}")
else()
  execute_process(
  	COMMAND bash -c "${CUDA_TOOLKIT_ROOT_DIR}/extras/demo_suite/deviceQuery | grep 'CUDA Capability' | sed -rn ':a;N;$!ba;s/.*:\\s*(.+)\\.(.+).*/sm_\\1\\2/p'"
    OUTPUT_VARIABLE GPU_ARCHITECTURE
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    )
  if("${GPU_ARCHITECTURE}" STREQUAL "")
    set(GPU_ARCHITECTURE "sm_60;sm_61;sm_70;sm_75")
    message("Could not detect CUDA device architecture, using ${GPU_ARCHITECTURE}")
  else()
    message("Detected CUDA architecture ${GPU_ARCHITECTURE}")
  endif()
endif()




# Build setup
# Do not edit

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")

find_package(GMP REQUIRED)


# Set CUDA compiler flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xptxas=-v -lineinfo --keep --compiler-options='-Wall  -Wno-unknown-pragmas'")

if(NOT "${GPU_MAX_REG}" STREQUAL "")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -maxrregcount=${GPU_MAX_REG}")
endif()

foreach(ARCH IN LISTS GPU_ARCHITECTURE)
    string(REPLACE "sm_" "compute_" COMPUTE ${ARCH})
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-code arch=${COMPUTE},code=${ARCH}")
endforeach(ARCH)


include(CTest)

# Generate version.h
include(${CMAKE_MODULE_PATH}/version.cmake)

# Generate build_config.h
configure_file(
        include/build_config.h.in
        ${CMAKE_CURRENT_BINARY_DIR}/generated/build_config.h
)

set(CMAKE_C_STANDARD 99)


# Concatenate all .cu files into a single kernel file for nvcc to
# work around the nvcc limitations of slow non-seperable builds.
# (nvcc does not know link-time optimization)
function(add_cuda_executable TARGET)
    set(CURRENT_C_SOURCES ${ARGN})
    set(CURRENT_CUDA_SOURCES ${ARGN})
    list(FILTER CURRENT_C_SOURCES EXCLUDE REGEX ".*\\.cu")
    list(FILTER CURRENT_CUDA_SOURCES INCLUDE REGEX ".*\\.cu")
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)
    add_custom_command(
            OUTPUT generated/${TARGET}-cudakernel.cu
            COMMAND cat ${CURRENT_CUDA_SOURCES} > ${CMAKE_CURRENT_BINARY_DIR}/generated/${TARGET}-cudakernel.cu
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            DEPENDS ${CURRENT_CUDA_SOURCES}
    )
    add_executable(${TARGET} ${CURRENT_C_SOURCES} generated/${TARGET}-cudakernel.cu)
endfunction()


# Set include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

find_package (Threads)
link_libraries(gmp)
link_libraries(${CMAKE_THREAD_LIBS_INIT})


# Set common source files
set(COMMON_CUDA_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mp/mp.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/mp/mp_montgomery.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecc/naf.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecc/tw_ed_common.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/ecm.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/factor_task.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/batch.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/stage1.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/stage2.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config/config.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config/handler.cu
        )

# Set coordinate specific ecc and ecm implementations
if(COORDINATES_EXTENDED)
  set(COMMON_CUDA_SOURCES
        ${COMMON_CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecc/tw_ed_extended.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/tw_ed_extended.cu
        )
elseif(COORDINATES_INVERTED)
  set(COMMON_CUDA_SOURCES
        ${COMMON_CUDA_SOURCES}
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecc/tw_ed_inverted.cu
        ${CMAKE_CURRENT_SOURCE_DIR}/src/ecm/tw_ed_inverted.cu
        )
endif()

set(COMMON_C_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/src/gmp_conv/gmp_conversion.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/config/ini.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/log.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/input/file.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/input/tcp.c
        ${CMAKE_CURRENT_SOURCE_DIR}/src/input/parser.c
        )

# Set output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(bench)
add_subdirectory(resource)

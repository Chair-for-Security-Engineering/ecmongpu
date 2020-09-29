add_cuda_executable(cuda-ecm
        ${COMMON_CUDA_SOURCES}
        ${COMMON_C_SOURCES}
        cuda-ecm.c
        )


add_executable(tcp-client
        tcp-client.c
        )

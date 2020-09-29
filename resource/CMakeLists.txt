set(RESOURCE_OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/res)
add_custom_target(resource_dir ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${RESOURCE_OUTPUT_DIR}/res)

configure_file(naf8192 ${RESOURCE_OUTPUT_DIR}/naf8192 COPYONLY)
configure_file(trpl4096 ${RESOURCE_OUTPUT_DIR}/trpl4096 COPYONLY)
configure_file(trpl-batch-4096 ${RESOURCE_OUTPUT_DIR}/trpl-batch-4096 COPYONLY)
configure_file(trpl-batch-8192 ${RESOURCE_OUTPUT_DIR}/trpl-batch-8192 COPYONLY)
configure_file(bouvier-8192 ${RESOURCE_OUTPUT_DIR}/bouvier-8192 COPYONLY)

configure_file(trpl-batch-50000 ${RESOURCE_OUTPUT_DIR}/trpl-batch-50000 COPYONLY)
configure_file(naf50000 ${RESOURCE_OUTPUT_DIR}/naf50000 COPYONLY)


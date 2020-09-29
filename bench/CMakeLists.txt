function(add_gpu_benchmark TARGET TESTFILE)
    set(TARGET "bench_gpu_${TARGET}")
    add_cuda_executable("${TARGET}"
            ${COMMON_C_SOURCES}
            ${COMMON_CUDA_SOURCES}
            ${TESTFILE}
            )
endfunction()

MACRO(SUBDIRLIST result curdir)
  FILE(GLOB children RELATIVE ${curdir} ${curdir}/*)
  SET(dirlist "")
  FOREACH(child ${children})
    IF(IS_DIRECTORY ${curdir}/${child})
      LIST(APPEND dirlist ${child})
    ENDIF()
  ENDFOREACH()
  SET(${result} ${dirlist})
ENDMACRO()


if (BUILD_BENCHMARKS)
    add_gpu_benchmark(bench_monprod bench_monprod.cu)

    find_package(PythonInterp)

    # Generate Config files
    configure_file(yield/generate.py
                   ${CMAKE_CURRENT_BINARY_DIR}/yield/generate.py)
    configure_file(yield/base.ini
	           ${CMAKE_CURRENT_BINARY_DIR}/yield/base.ini COPYONLY)
    configure_file(yield/parse.py
	    	   ${CMAKE_CURRENT_BINARY_DIR}/yield/parse.py COPYONLY)
    configure_file(yield/print.py
	    	   ${CMAKE_CURRENT_BINARY_DIR}/yield/print.py COPYONLY)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} generate.py
		    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/yield/)

    # Copy over input files
    file(GLOB inputfiles "yield/input/*.input")
    file(COPY ${inputfiles}
	    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/yield/input/)

    # Add target for stage one benchmarks
    SUBDIRLIST(benchmarks "${CMAKE_CURRENT_BINARY_DIR}/yield/configs/")
    foreach(case ${benchmarks})
    	file(GLOB configs "${CMAKE_CURRENT_BINARY_DIR}/yield/configs/${case}/*.ini")
    	foreach(config ${configs})
    	    get_filename_component(BASENAME ${config} NAME_WE)
    		add_custom_command(
			COMMENT "Running benchmark ${case}/${BASENAME}"
    	        	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/yield/log/${case}/${BASENAME}
    	        	COMMAND cuda-ecm -c ${config} > /dev/null
    	        )
		add_custom_target(bench_${case}_${BASENAME}
    	        	COMMAND cuda-ecm -c ${config} > /dev/null
			)
    		add_custom_command(
    	        	OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/yield/report.${case}.${BASENAME}
    	        	COMMAND ${PYTHON_EXECUTABLE} parse.py ${case} 
    	        		${CMAKE_CURRENT_BINARY_DIR}/yield/log/${case}/${BASENAME} 
    	    		${CMAKE_CURRENT_BINARY_DIR}/yield/output/${case}/${BASENAME}
    	    		${CMAKE_CURRENT_BINARY_DIR}/yield/report.${case}.${BASENAME}
    	        	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/yield/
    	        	DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/yield/log/${case}/${BASENAME}
    	        )
    	    list(APPEND benchlogs ${CMAKE_CURRENT_BINARY_DIR}/yield/report.${case}.${BASENAME})
    	endforeach()

    endforeach()

    list(SORT benchlogs)

    add_custom_target(bench
	    COMMAND cat ${benchlogs} > ${CMAKE_CURRENT_BINARY_DIR}/yield/report
	    COMMAND ${PYTHON_EXECUTABLE} print.py ${CMAKE_CURRENT_BINARY_DIR}/yield/report
    	    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/yield/
	    DEPENDS ${benchlogs}
	    )

endif ()

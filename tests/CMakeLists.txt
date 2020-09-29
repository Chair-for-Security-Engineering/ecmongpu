function(add_cpu_test TARGET TESTFILE)
    set(TARGET "test_cpu_${TARGET}")
    add_cuda_executable("${TARGET}"
            ${COMMON_C_SOURCES}
            ${COMMON_CUDA_SOURCES}
            ${TESTFILE}
            )
    add_test(NAME ${TARGET} COMMAND ${TARGET})
    set_tests_properties(${TARGET} PROPERTIES
            PASS_REGULAR_EXPRESSION "PASSED"
            TIMEOUT 20)
endfunction()

function(add_gpu_test TARGET TESTFILE)
    set(TARGET "test_gpu_${TARGET}")
    add_cuda_executable("${TARGET}"
            ${COMMON_C_SOURCES}
            ${COMMON_CUDA_SOURCES}
            ${TESTFILE}
            )
    add_test(NAME ${TARGET} COMMAND ${TARGET})
    set_tests_properties(${TARGET} PROPERTIES
            PASS_REGULAR_EXPRESSION "PASSED"
            TIMEOUT 20)
endfunction()


if (BUILD_TESTING)

    # CPU Tests
    add_cpu_test(mp_conversion cpu/mp/test_mp_conversion.c)
    add_cpu_test(mp_sub cpu/mp/test_mp_sub.c)
    add_cpu_test(mp_sub_mod cpu/mp/test_mp_sub_mod.c)
    add_cpu_test(mp_cmp cpu/mp/test_mp_cmp.c)
    add_cpu_test(mp_gt cpu/mp/test_mp_gt.c)
    add_cpu_test(mp_add cpu/mp/test_mp_add.c)
    add_cpu_test(mp_mul_ui cpu/mp/test_mp_mul_ui.c)
    add_cpu_test(mp_mul cpu/mp/test_mp_mul.c)
    add_cpu_test(mp_srlimbs cpu/mp/test_mp_srlimbs.c)
    add_cpu_test(mp_testbit cpu/mp/test_mp_testbit.c)
    add_cpu_test(mp_setbit cpu/mp/test_mp_setbit.c)

    add_cpu_test(mp_montgomery_montrans cpu/mp_montgomery/test_mp_montrans.c)
    add_cpu_test(mp_montgomery_monprod cpu/mp_montgomery/test_mp_monprod.c)
    add_cpu_test(mp_montgomery_monexp cpu/mp_montgomery/test_mp_monexp.c)
    add_cpu_test(mp_montgomery_moninv cpu/mp_montgomery/test_mp_moninv.c)

    add_cpu_test(ecc_arithm cpu/ecc/test_ecc.c)
    add_cpu_test(ecc_point_on_curve cpu/ecc/test_point_on_curve.c)

    add_cpu_test(naf cpu/naf/test_naf.c)

    add_cpu_test(factor_list_sorted cpu/ecm/test_factor_list_sort.c)

    # GPU Tests
    add_gpu_test(mp_mul gpu/mp/test_mp_mul.cu)
    add_gpu_test(mp_copy gpu/mp/test_mp_copy.cu)
    add_gpu_test(mp_add gpu/mp/test_mp_add.cu)
    add_gpu_test(mp_sub gpu/mp/test_mp_sub.cu)
    add_gpu_test(mp_sub_mod gpu/mp/test_mp_sub_mod.cu)
    add_gpu_test(mp_gt gpu/mp/test_mp_gt.cu)
    add_gpu_test(mp_cmp gpu/mp/test_mp_cmp.cu)

    add_gpu_test(mp_montgomery_monprod gpu/mp_montgomery/test_mp_monprod.cu)
    add_gpu_test(mp_montgomery_monprod_distinct gpu/mp_montgomery/test_mp_monprod_distinct.cu)
    add_gpu_test(mp_montgomery_monsquare gpu/mp_montgomery/test_mp_monsquare.cu)
    add_gpu_test(mp_montgomery_monexp gpu/mp_montgomery/test_mp_monexp.cu)
    add_gpu_test(mp_montgomery_moninv gpu/mp_montgomery/test_mp_moninv.cu)

    add_gpu_test(ecc_arithm gpu/ecc/test_ecc.cu)
    add_gpu_test(ecc_point_on_curve gpu/ecc/test_point_on_curve.cu)

    add_gpu_test(naf_smul gpu/ecm/test_naf_smul.cu)
    set_tests_properties(test_gpu_naf_smul PROPERTIES TIMEOUT ${BITWIDTH})

    add_gpu_test(naf_smul_nafarr gpu/ecm/test_naf_smul_nafarr.cu)
    set_tests_properties(test_gpu_naf_smul_nafarr PROPERTIES TIMEOUT ${BITWIDTH})

    add_gpu_test(trplchain_smul gpu/ecm/test_trplchain_smul.cu)
    set_tests_properties(test_gpu_trplchain_smul PROPERTIES TIMEOUT ${BITWIDTH})

    add_gpu_test(trpl_batch_smul gpu/ecm/test_trpl_batch_smul.cu)
    set_tests_properties(test_gpu_trpl_batch_smul PROPERTIES TIMEOUT ${BITWIDTH})

    add_gpu_test(batch_bouvier8192 gpu/ecm/test_bouvier8192.cu)
    set_tests_properties(test_gpu_batch_bouvier8192 PROPERTIES TIMEOUT ${BITWIDTH})


endif ()

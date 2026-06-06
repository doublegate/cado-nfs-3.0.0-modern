add_executable(check_rels check_rels.cpp)
target_link_libraries(check_rels utils_with_io)
install(TARGETS check_rels RUNTIME DESTINATION ${LIBSUFFIX}/misc)

if (HAVE_GMPECM)
add_executable(descent_init_Fp descent_init_Fp.cpp smooth_detect.cpp)
target_link_libraries(descent_init_Fp utils ecm)
install(TARGETS descent_init_Fp RUNTIME DESTINATION ${LIBSUFFIX}/misc)
endif()

add_executable(debug_renumber EXCLUDE_FROM_ALL debug_renumber.cpp)
target_link_libraries(debug_renumber utils)

add_executable(explain_indexed_relation EXCLUDE_FROM_ALL
    explain_indexed_relation.cpp)
target_link_libraries(explain_indexed_relation utils)

add_executable(indexed_relations_to_density EXCLUDE_FROM_ALL indexed_relations_to_density.cpp)
target_link_libraries(indexed_relations_to_density utils_with_io)

mark_targets_as_openmp(indexed_relations_to_density)

if(HAVE_GMPECM)
    add_executable(factor factor.cpp)
    target_link_libraries(factor utils ecm)
endif()

add_executable(cl_structure_from_group_order cl_structure_from_group_order.cpp)
target_link_libraries(cl_structure_from_group_order utils)

# GPU ECM pre-NFS factoring front-end (v3.1.0-modern, Track 2.1). Strips small/
# medium factors from N on the GPU before NFS. Built only when the CUDA GPU
# backend is enabled (-DENABLE_GPU=ON, which sets HAVE_GPU_ECM and enables the
# CUDA language in config/cuda.cmake). See misc/gpu_prefactor/README.md.
if(HAVE_GPU_ECM)
    add_executable(gpu-prefactor gpu_prefactor/gpu-prefactor.cu)
    # Default to the reference RTX 3090 (sm_86); override -DCADO_GPU_ARCH=<cc>.
    # (CMake 4.x can auto-pick a stale CMAKE_CUDA_ARCHITECTURES like 75.)
    if(NOT DEFINED CADO_GPU_ARCH)
        set(CADO_GPU_ARCH 86)
    endif()
    set_target_properties(gpu-prefactor PROPERTIES CUDA_ARCHITECTURES "${CADO_GPU_ARCH}")
    # CADO sets no CMAKE_BUILD_TYPE (it uses local.sh flags), so the default CUDA
    # flags carry no optimization; force device -O3 or the kernels run ~30x slow.
    target_compile_options(gpu-prefactor PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    target_include_directories(gpu-prefactor PRIVATE ${GMP_INCDIR})
    target_link_libraries(gpu-prefactor CUDA::cudart ${gmp_libname})
    install(TARGETS gpu-prefactor RUNTIME DESTINATION ${LIBSUFFIX}/misc)
endif()

include(CMakeLists-nodist.txt OPTIONAL)

cmake_minimum_required(VERSION 3.15)
project(algorithms)

include_directories(${HMM_FASTA_processing_SOURCE_DIR}/data_readers)

set(ALGORITHMS_SRC MSV_HMM.hpp MSV_HMM.cpp MSV_kernel_store.hpp)

function(add_library_with_sycl)
    set(one_value_args TARGET)
    set(multi_value_args SOURCES SYCL_SOURCES)
    cmake_parse_arguments(ADD_LIBRARY
            "${options}"
            "${one_value_args}"
            "${multi_value_args}"
            ${ARGN}
            )
    add_library(${ADD_LIBRARY_TARGET} ${ADD_LIBRARY_SOURCES})
    add_sycl_to_target(TARGET ${ADD_LIBRARY_TARGET}
            SOURCES ${ADD_LIBRARY_SYCL_SOURCES})
endfunction(add_library_with_sycl)

add_library_with_sycl(TARGET algorithms SOURCES ${ALGORITHMS_SRC} SYCL_SOURCES MSV_HMM.cpp)

add_executable(test_MSV test_MSV.cpp ${ALGORITHMS_SRC})
target_link_libraries(test_MSV algorithms)
target_link_libraries(test_MSV data_readers)
target_link_libraries(test_MSV stdc++fs)

add_test(test_MSV test_MSV)

add_executable(benchmark_MSV benchmark_MSV.cpp ${ALGORITHMS_SRC})
target_link_libraries(benchmark_MSV algorithms)
target_link_libraries(benchmark_MSV data_readers)
target_link_libraries(benchmark_MSV stdc++fs)

add_executable(benchmark_MSV_1400 benchmark_MSV_1400.cpp ${ALGORITHMS_SRC})
target_link_libraries(benchmark_MSV_1400 algorithms)
target_link_libraries(benchmark_MSV_1400 data_readers)


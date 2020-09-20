cmake_minimum_required(VERSION 3.15)
project(algorithms)

set(ALGORITHMS_SRC MSV_HMM.hpp MSV_HMM.cpp)

add_library(algorithms ${ALGORITHMS_SRC})

add_executable(test_MSV test_MSV.cpp)
target_link_libraries(test_MSV algorithms)
target_link_libraries(test_MSV data_readers)
target_link_libraries(test_MSV stdc++fs)

add_test(test_MSV test_MSV)

add_executable(benchmark_MSV benchmark_MSV.cpp)
target_link_libraries(benchmark_MSV algorithms)
target_link_libraries(benchmark_MSV data_readers)
target_link_libraries(benchmark_MSV stdc++fs)

add_executable(benchmark_MSV_1400 benchmark_MSV_1400.cpp)
target_link_libraries(benchmark_MSV_1400 algorithms)
target_link_libraries(benchmark_MSV_1400 data_readers)

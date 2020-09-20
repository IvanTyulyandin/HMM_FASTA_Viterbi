cmake_minimum_required(VERSION 3.15)
project(HMM_FASTA_processing)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-unused-function -Wno-unused-variable)
add_compile_options(-march=native -O3 -pipe)

enable_testing()

include_directories(${HMM_FASTA_processing_SOURCE_DIR}/data_readers)
include_directories(${HMM_FASTA_processing_SOURCE_DIR}/algorithms)

add_subdirectory(data_readers)
add_subdirectory(algorithms)

add_executable(HMM_FASTA_Viterbi main.cpp)

target_link_libraries(HMM_FASTA_Viterbi data_readers)
target_link_libraries(HMM_FASTA_Viterbi algorithms)

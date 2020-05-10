cmake_minimum_required(VERSION 3.15)
project(HMM_FASTA_processing)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
find_package(ComputeCpp REQUIRED)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-unused-function -Wno-unused-variable)

add_subdirectory(data_models)

include_directories(${HMM_FASTA_processing_SOURCE_DIR}/data_models)

add_executable(HMM_FASTA_Viterbi main.cpp)

target_link_libraries(HMM_FASTA_Viterbi data_models)
target_link_libraries(HMM_FASTA_Viterbi data_models_test)

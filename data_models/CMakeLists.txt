cmake_minimum_required(VERSION 3.15)
project(data_models)

set(DATA_MODELS_SRC
        Profile_HMM.h Profile_HMM.cpp
        FASTA_protein_sequences.h FASTA_protein_sequences.cpp
        MSV_HMM.h MSV_HMM.cpp)

set(DATA_MODELS_TEST_SRC test_hmm_parsing.h test_fasta_parsing.h test_MSV.h)

add_library(data_models ${DATA_MODELS_SRC})
add_library(data_models_test ${DATA_MODELS_TEST_SRC})

set(CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)
include(FindComputeCpp)
include_directories(${COMPUTECPP_INCLUDE_DIRECTORY})

target_link_libraries(data_models ComputeCpp::ComputeCpp)

target_link_libraries(data_models_test data_models)

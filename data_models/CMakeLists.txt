cmake_minimum_required(VERSION 3.15)
project(data_models)

set(DATA_MODELS_SRC
        Profile_HMM.hpp Profile_HMM.cpp
        FASTA_protein_sequences.hpp FASTA_protein_sequences.cpp
        MSV_HMM.hpp MSV_HMM.cpp)

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

add_library_with_sycl(TARGET data_models SOURCES ${DATA_MODELS_SRC} SYCL_SOURCES MSV_HMM.cpp)

add_executable(test_hmm_parsing test_hmm_parsing.cpp ${DATA_MODELS_SRC})
target_link_libraries(test_hmm_parsing data_models)

add_executable(test_fasta_parsing test_fasta_parsing.cpp ${DATA_MODELS_SRC})
target_link_libraries(test_fasta_parsing data_models)

add_executable(test_MSV test_MSV.cpp ${DATA_MODELS_SRC})
target_link_libraries(test_MSV data_models)

add_test(hmm_parsing_test test_hmm_parsing)
add_test(fasta_parsing_test test_fasta_parsing)
add_test(MSV_test test_MSV)

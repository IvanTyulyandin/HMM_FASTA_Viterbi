cmake_minimum_required(VERSION 3.15)
project(data_models)

set(DATA_MODELS_SRC
        Profile_HMM.h Profile_HMM.cpp
        FASTA_protein_sequences.h FASTA_protein_sequences.cpp
        MSV_HMM.h MSV_HMM.cpp)

set(DATA_MODELS_TEST_SRC test_hmm_parsing.h test_fasta_parsing.h test_MSV.h)

function(add_library_with_sycl)
  set(one_value_args TARGET)
  set(multi_value_args SOURCES)
  cmake_parse_arguments(ADD_LIBRARY
    "${options}"
    "${one_value_args}"
    "${multi_value_args}"
    ${ARGN}
  )
  add_library(${ADD_LIBRARY_TARGET} ${ADD_LIBRARY_SOURCES})
  add_sycl_to_target(TARGET ${ADD_LIBRARY_TARGET}
    SOURCES ${ADD_LIBRARY_SOURCES})
endfunction(add_library_with_sycl)

add_library_with_sycl(TARGET data_models SOURCES ${DATA_MODELS_SRC})

add_library(data_models_test ${DATA_MODELS_TEST_SRC})
target_link_libraries(data_models_test data_models)

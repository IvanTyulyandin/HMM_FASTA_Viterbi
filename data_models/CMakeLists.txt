cmake_minimum_required(VERSION 3.15)
project(data_models)

set(DATA_MODELS_SRC hmm.h hmm.cpp fasta_protein_sequences.h fasta_protein_sequences.cpp)
set(DATA_MODELS_TEST_SRC test_hmm_parsing.h test_fasta_parsing.h)

add_library(data_models ${DATA_MODELS_SRC})
add_library(data_models_test ${DATA_MODELS_SRC} ${DATA_MODELS_TEST_SRC})

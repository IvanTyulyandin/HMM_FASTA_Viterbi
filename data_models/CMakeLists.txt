cmake_minimum_required(VERSION 3.15)
project(data_models)

set(DATA_MODELS_SRC hmm.h hmm.cpp fasta_protein_sequences.h fasta_protein_sequences.cpp)

add_library(data_models ${DATA_MODELS_SRC})

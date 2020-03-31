cmake_minimum_required(VERSION 3.15)
project(HMM_FASTA_processing)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall -Wextra -pedantic -Werror -Wno-unused-function)

add_executable(HMM_FASTA_Viterbi main.cpp hmm.h hmm.cpp fasta_protein_sequences.h fasta_protein_sequences.cpp)

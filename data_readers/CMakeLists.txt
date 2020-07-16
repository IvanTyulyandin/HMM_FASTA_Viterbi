cmake_minimum_required(VERSION 3.15)
project(data_readers)

set(HMM_PARSER_SRC Profile_HMM.hpp Profile_HMM.cpp)
set(FASTA_PARSER_SRC FASTA_protein_sequences.hpp FASTA_protein_sequences.cpp)

add_library(data_readers ${HMM_PARSER_SRC} ${FASTA_PARSER_SRC})

add_executable(test_hmm_parsing test_hmm_parsing.cpp ${HMM_PARSER_SRC})
target_link_libraries(test_hmm_parsing data_readers)
target_link_libraries(test_hmm_parsing stdc++fs)

add_executable(test_fasta_parsing test_fasta_parsing.cpp ${FASTA_PARSER_SRC})
target_link_libraries(test_fasta_parsing data_readers)
target_link_libraries(test_fasta_parsing stdc++fs)

add_test(test_hmm_parsing test_hmm_parsing)
add_test(test_fasta_parsing_test test_fasta_parsing)

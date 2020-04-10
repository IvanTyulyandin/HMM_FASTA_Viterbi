# Viterbi algorithms
This is an implementation of different bioinformatics algorithms,
based on Viterbi algorithm. All of them takes .hmm file (Hidden Markov Model)
and sequence of proteins in FASTA format.

### How to build
You can use (or modify) ```compile_clang_in_build_dir.sh``` and ```run_in_build_dir.sh```.
As it follows from namings, 
first script compiles the project with clang in the build directory,
and second one runs executable at that directory.

Otherwise use ```cmake . && make && ./HMM_FASTA_processing```.

### TODO
* SSV (Single Segment Viterbi)
* MSV (Multiple Segment Viterbi)
* Viterbi
# Viterbi algorithms
This is an implementation of different bioinformatics algorithms,
based on Viterbi algorithm. All of them takes .hmm file (Hidden Markov Model)
and sequence of proteins in FASTA format.

### Prerequisites
OpenCL and [ComputeCpp SYCL compiler](https://developer.codeplay.com/products/computecpp/ce/home/), Community Edition.

### How to build
You can use (or modify) ```compile_clang_in_build_dir.sh``` and ```run_in_build_dir.sh```.
In ```compile_clang_in_build_dir.sh``` you should change ```-DComputeCpp_DIR``` to the path where ComputeCpp is located.
As it follows from namings, first script compiles the project with clang in the build directory,
and second one runs executable at that directory.


# HMM_FASTA_Viterbi
This is an implementation of different bioinformatics algorithms for protein analysis, based on the Viterbi algorithm.
The main purpose is to rewrite some functionality from the [HMMer project](https://github.com/EddyRivasLab/hmmer) for devices with OpenCL support, such as GPGPUs.
Each algorithm takes .hmm file (Hidden Markov Model) and sequence of proteins in FASTA format as parameters.
More about these formats can be found at [HMMer Userguide pdf](http://eddylab.org/software/hmmer/Userguide.pdf), page 210 for .hmm and page 224 for .fasta.

### Prerequisites
OpenCL and [ComputeCpp SYCL compiler](https://developer.codeplay.com/products/computecpp/ce/home/), Community Edition.


### How to build
You can use (or modify) ```compile_clang_in_build_dir.sh``` and ```run_in_build_dir.sh```.
In ```compile_clang_in_build_dir.sh``` you should change ```-DComputeCpp_DIR``` to the path where ComputeCpp is located.
As it follows from namings, first script compiles the project with clang in the build directory,
and second one runs executable at that directory.


### Status
Work in progress. There is no code inside ```main.cpp```, but some unit tests can be run with ```ctest```.
In ```data_readers``` directory there are partial parsers for .hmm and .fasta formats.
The implementation of MSV (Multiple Segment Viterbi) algorithm is in directory ```algorithms```.


#!/bin/bash

mkdir -p build
cp *.hmm build/
cp *.fsa build/
cd build
cmake -DCMAKE_BUILD_TYPE=Debug -D CMAKE_C_COMPILER=clang -D CMAKE_CXX_COMPILER=clang++ ..
make -j 3
cd ..

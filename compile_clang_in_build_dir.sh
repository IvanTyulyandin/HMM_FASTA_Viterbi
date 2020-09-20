#!/bin/bash

mkdir -p build
cp -u -r profile_HMMs build/
cp -u -r FASTA_files build/
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    ..
make -j 3
cd ..

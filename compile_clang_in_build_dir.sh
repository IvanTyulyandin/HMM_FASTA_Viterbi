#!/bin/bash

mkdir -p build
cp -u -r profile_HMMs build/
cp *.fsa build/
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DComputeCpp_DIR=/usr/local/ComputeCpp-CE-2.0.0-x86_64-linux-gnu/ \
..
make -j 3
cd ..

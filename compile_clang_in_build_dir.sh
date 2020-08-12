#!/bin/bash

cd algorithms
cp /dev/null MSV_HMM_spec.cpp
cp /dev/null MSV_HMM_spec.hpp
python3 MSV_p_gen.py $(find ../profile_HMMs -maxdepth 1 -type f -name "*.hmm")
cd ..

mkdir -p build
cp -u -r profile_HMMs build/
cp *.fsa build/
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DComputeCpp_DIR=/usr/local/ComputeCpp-CE-2.1.0-x86_64-linux-gnu/ \
..
make -j 3
cd ..

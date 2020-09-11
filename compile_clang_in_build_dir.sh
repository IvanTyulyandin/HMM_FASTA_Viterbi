#!/bin/bash

cd algorithms
python3 clean_MSV_kernel_store.py
if [ "$1" != "only_one" ];
then
    python3 MSV_p_gen.py $(find ../profile_HMMs -maxdepth 1 -type f -name "*.hmm")
else
    python3 MSV_p_gen.py ../profile_HMMs/1400.hmm
fi
cd ..

mkdir -p build
cp -u -r profile_HMMs build/
cp -u -r FASTA_files build/
cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DComputeCpp_DIR=/usr/local/ComputeCpp-CE-2.1.0-x86_64-linux-gnu/ \
    ..
make -j 3
cd ..

if [ "$1" != "only_one" ];
then
    echo "Code was specialized with all HMMs from folder profile_HMMs"
else
    echo "Code was specialized with 1400.hmm, only benchmark_MSV_1400 will be working"
fi

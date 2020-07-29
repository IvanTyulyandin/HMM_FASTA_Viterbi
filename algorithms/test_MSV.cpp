#include "FASTA_protein_sequences.hpp"
#include "MSV_HMM.hpp"

#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>

// code below was taken from
// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp = 5) {
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
           // unless the result is subnormal
           || std::fabs(x - y) < std::numeric_limits<T>::min();
}

int main() {
    // Test MSV checking invariant for sequential and parallel MSV implementations
    namespace fs = std::experimental::filesystem;
    auto fasta = FASTA_protein_sequences("../fasta_like_example.fsa");

    for (const auto& profile : fs::directory_iterator("../profile_HMMs")) {
        if (profile.path().extension() == ".hmm") {
            auto msv = MSV_HMM(Profile_HMM(profile.path()));
            for (const auto& protein : fasta.sequences) {
                assert(almost_equal(msv.run_on_sequence(protein), msv.parallel_run_on_sequence(protein)));
            }
        }
    }

    return 0;
}
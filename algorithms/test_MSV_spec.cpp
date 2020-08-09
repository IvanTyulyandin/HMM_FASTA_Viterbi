#include "FASTA_protein_sequences.hpp"
#include "MSV_HMM.hpp"
#include "MSV_HMM_spec.hpp"

#include <cassert>
#include <cmath>
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
    // Test MSV_spec checking invariant for parallel and specialized MSV implementations
    auto fasta = FASTA_protein_sequences("../fasta_like_example.fsa");
    auto spec_HMM_path = std::string("../profile_HMMs/100.hmm");
    auto msv = MSV_HMM(Profile_HMM(spec_HMM_path));

    for (const auto& seq : fasta.sequences) {
        auto par = msv.parallel_run_on_sequence(seq);
        auto spec = MSV_HMM_spec::parallel_run_on_sequence_100(seq);
        std::cout << par << ' ' << spec << '\n';
        assert(almost_equal(par, spec));
    }
    return 0;
}
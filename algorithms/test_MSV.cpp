#include "FASTA_protein_sequences.hpp"
#include "MSV_HMM.hpp"

#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y) {
    return std::fabs(x - y) <= 0.0001;
}

int main() {
    // Test MSV checking invariant for sequential and parallel MSV implementations
    namespace fs = std::experimental::filesystem;
    auto fasta = FASTA_protein_sequences("../FASTA_files/fasta_like_example.fsa");

    for (const auto& profile : fs::directory_iterator("../profile_HMMs")) {
        if (profile.path().extension() == ".hmm") {
            auto msv = MSV_HMM(Profile_HMM(profile.path()));
            for (const auto& protein : fasta.sequences) {
                auto seq = msv.run_on_sequence(protein);
                auto par = msv.parallel_run_on_sequence(protein);
                auto par_spec = msv.parallel_run_on_sequence(protein, true);
                if (!almost_equal(seq, par) || !almost_equal(par, par_spec)) {
                    std::cout << "test_MSV failed!\n"
                         << "Seq: " << seq << ", par " << par << ", par_spec " << par_spec << '\n';
                    exit(1);
                }
            }
        }
    }

    return 0;
}

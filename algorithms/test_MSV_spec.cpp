#include "MSV_HMM.hpp"
#include "MSV_HMM_spec.hpp"

#include <cassert>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>
#include <unordered_map>

template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y) {
    return std::fabs(x - y) <= 0.0001;
}

int main() {
    // Test MSV_spec checking invariant for parallel and specialized MSV implementations

    using MSV_method_ptr = Log_score(*)(const Protein_sequence&);
    auto methods = std::unordered_map<std::string, MSV_method_ptr>({
        {"100", &MSV_HMM_spec::parallel_run_on_sequence_100},
        {"200", &MSV_HMM_spec::parallel_run_on_sequence_200},
        {"300", &MSV_HMM_spec::parallel_run_on_sequence_300},
        {"400", &MSV_HMM_spec::parallel_run_on_sequence_400},
        {"500", &MSV_HMM_spec::parallel_run_on_sequence_500},
        {"600", &MSV_HMM_spec::parallel_run_on_sequence_600},
        {"700", &MSV_HMM_spec::parallel_run_on_sequence_700},
        {"800", &MSV_HMM_spec::parallel_run_on_sequence_800},
        {"900", &MSV_HMM_spec::parallel_run_on_sequence_900},
        {"1001", &MSV_HMM_spec::parallel_run_on_sequence_1001},
        {"1100", &MSV_HMM_spec::parallel_run_on_sequence_1100},
        {"1200", &MSV_HMM_spec::parallel_run_on_sequence_1200},
        {"1301", &MSV_HMM_spec::parallel_run_on_sequence_1301},
        {"1400", &MSV_HMM_spec::parallel_run_on_sequence_1400},
        {"1509", &MSV_HMM_spec::parallel_run_on_sequence_1509},
        {"1600", &MSV_HMM_spec::parallel_run_on_sequence_1600},
        {"1705", &MSV_HMM_spec::parallel_run_on_sequence_1705},
        {"1799", &MSV_HMM_spec::parallel_run_on_sequence_1799},
        {"1901", &MSV_HMM_spec::parallel_run_on_sequence_1901},
        {"2050", &MSV_HMM_spec::parallel_run_on_sequence_2050},
        {"2138", &MSV_HMM_spec::parallel_run_on_sequence_2138},
        {"2207", &MSV_HMM_spec::parallel_run_on_sequence_2207},
        {"2365", &MSV_HMM_spec::parallel_run_on_sequence_2365},
        {"2405", &MSV_HMM_spec::parallel_run_on_sequence_2405}
    });

    namespace fs = std::experimental::filesystem;
    auto fasta = FASTA_protein_sequences("../fasta_like_example.fsa");

    for (const auto& profile : fs::directory_iterator("../profile_HMMs")) {
        if (profile.path().extension() == ".hmm") {
            auto msv = MSV_HMM(Profile_HMM(profile.path()));
            auto spec_ptr = methods[profile.path().stem()];
            for (const auto& protein : fasta.sequences) {
                auto par = msv.parallel_run_on_sequence(protein);
                auto spec = (*spec_ptr)(protein);
                assert(almost_equal(par, spec));
            }
        }
    }
    return 0;
}

#include "benchmark_helper.hpp"

constexpr size_t TIMES_TO_RUN = 2;

int main() {
    // benchmark_MSV_1400 runs parallel MSV with 1400.hmm and FASTA

    auto fasta = FASTA_protein_sequences("../FASTA_files/random_FASTA.fsa");

    auto msv = MSV_HMM(Profile_HMM("../profile_HMMs/1400.hmm"));

    benchmark_N_times<TIMES_TO_RUN>(fasta, msv, Algorithm_selector::par, "Parallel MSV 1400.hmm");
    benchmark_N_times<TIMES_TO_RUN>(fasta, msv, Algorithm_selector::par_spec, "Specialized MSV 1400.hmm");

    return 0;
}

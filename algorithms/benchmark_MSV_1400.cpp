#include "MSV_HMM.hpp"

#include <iostream>
#include <chrono>


int main() {
    // benchmark_MSV_1400 runs parallel MSV with 1400.hmm and FASTA

    auto fasta = FASTA_protein_sequences("../FASTA_files/random_FASTA.fsa");

    auto all_time = std::chrono::milliseconds{0};

    auto msv = MSV_HMM(Profile_HMM("../profile_HMMs/1400.hmm"));

    for (const auto& protein : fasta.sequences) {
        auto start_time = std::chrono::steady_clock::now();

        msv.parallel_run_on_sequence(protein);

        auto cur = std::chrono::steady_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cur - start_time);
        all_time += duration;
    }
    std::cout << "MSV_1400: " << all_time.count() << " milliseconds\n";
    return 0;
}

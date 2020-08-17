#include "MSV_HMM.hpp"

#include <experimental/filesystem>
#include <iostream>
#include <chrono>


int main() {
    // benchmark_MSV runs parallel MSV implementation against big FASTA batch

    namespace fs = std::experimental::filesystem;
    auto fasta = FASTA_protein_sequences("../FASTA_files/random_FASTA.fsa");

    auto all_time = std::chrono::milliseconds{0};

    for (const auto& profile : fs::directory_iterator("../profile_HMMs")) {
        if (profile.path().extension() == ".hmm") {
            auto file_name = profile.path().stem();
            auto msv = MSV_HMM(Profile_HMM(profile.path()));

            for (const auto& protein : fasta.sequences) {
                auto start_time = std::chrono::steady_clock::now();

                msv.parallel_run_on_sequence(protein);

                auto cur = std::chrono::steady_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cur - start_time);
                all_time += duration;
                std::cout << file_name << ": " << duration.count() << " msec\n";
            }
        }
    }
    std::cout << "Elapsed time is " << all_time.count() << " milliseconds\n";
    return 0;
}

#include "MSV_HMM_spec.hpp"

#include <experimental/filesystem>
#include <unordered_map>
#include <iostream>
#include <chrono>


int main() {
    // benchmark_MSV_spec runs specialized MSV implementation against big FASTA batch

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
    auto fasta = FASTA_protein_sequences("../FASTA_files/random_FASTA.fsa");

    auto all_time = std::chrono::milliseconds{0};

    for (const auto& profile : fs::directory_iterator("../profile_HMMs")) {
        if (profile.path().extension() == ".hmm") {
            auto file_name = profile.path().stem();
            auto spec_ptr = methods.find(file_name);

            for (const auto& protein : fasta.sequences) {
                auto start_time = std::chrono::steady_clock::now();

                (spec_ptr->second)(protein);

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

#include "benchmark_helper.hpp"

#include <experimental/filesystem>
#include <iostream>
#include <chrono>

using MSV_file_name = std::string;
using MSV_descriptor = std::vector<std::pair<MSV_HMM, MSV_file_name>>;

constexpr size_t TIMES_TO_RUN = 2;

void benchmark_MSV(
    const FASTA_protein_sequences& fasta,
    MSV_descriptor& msv_vector,
    const Algorithm_selector selector,
    const std::string& algo_description) 
{
    auto all_time = std::chrono::milliseconds{0};
    for (auto& [msv, name] : msv_vector) {
        auto message = algo_description + ' ' + name;
        all_time += benchmark_N_times<TIMES_TO_RUN>(fasta, msv, selector, message);
    }
    std::cout << '\n' << algo_description << " best times sum: " << all_time.count() << " milliseconds\n\n";
}

int main() {
    namespace fs = std::experimental::filesystem;
    auto fasta = FASTA_protein_sequences("../FASTA_files/random_FASTA.fsa");

    auto msv_vector = MSV_descriptor{};

    for (const auto& profile : fs::directory_iterator("../profile_HMMs")) {
        if (profile.path().extension() == ".hmm") {
            auto file_name = profile.path().stem();
            msv_vector.emplace_back(
                std::make_pair(Profile_HMM(profile.path()), file_name));
        }
    }

    benchmark_MSV(fasta, msv_vector, Algorithm_selector::par, "Parallel MSV");
    benchmark_MSV(fasta, msv_vector, Algorithm_selector::par_spec, "Specialized MSV");
    return 0;
}

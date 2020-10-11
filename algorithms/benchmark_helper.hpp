#include "MSV_HMM.hpp"

#include <iostream>
#include <chrono>

enum class Algorithm_selector { seq, par, par_spec };

template<int N>
std::chrono::milliseconds benchmark_N_times(
    const FASTA_protein_sequences& fasta, 
    MSV_HMM& msv,
    const Algorithm_selector algorithm_selector,
    std::string_view message)
{
    auto best_time = std::chrono::milliseconds::max();

    for (size_t i = 0; i < N; ++i) {
        auto cur_iter_time = std::chrono::milliseconds{0};
        for (const auto& protein : fasta.sequences) {
            auto iteration_start_time = std::chrono::steady_clock::now();
            switch (algorithm_selector) {
                case Algorithm_selector::seq:
                    msv.run_on_sequence(protein);
                    break;
                case Algorithm_selector::par:
                    msv.parallel_run_on_sequence(protein, false);
                    break;
                case Algorithm_selector::par_spec:
                    msv.parallel_run_on_sequence(protein, true);
                    break;
                default:
                    std::cout << "Unknown algorithm selector\n";
                    break;
            }

            auto cur_time = std::chrono::steady_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(cur_time - iteration_start_time);
            cur_iter_time += duration;
        }
        best_time = std::min(best_time, cur_iter_time);
    }
    std::cout << message << ": best time is " << best_time.count() << " msec from " << N << " times \n";
    return best_time;
}

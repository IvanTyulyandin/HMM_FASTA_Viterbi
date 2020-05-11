#include "MSV_HMM.h"

#include <CL/sycl.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <iostream>

constexpr float minus_infinity = -std::numeric_limits<float>::infinity();

namespace {

    // Default background frequencies for protein models.
    // Numbers were taken from p7_AminoFrequencies hmmer function
    constexpr auto background_frequencies = std::array<float, NUM_OF_AMINO_ACIDS> {
            0.0787945, 0.0151600, 0.0535222, 0.0668298, // A C D E
            0.0397062, 0.0695071, 0.0229198, 0.0590092, // F G H I
            0.0594422, 0.0963728, 0.0237718, 0.0414386, // K L M N
            0.0482904, 0.0395639, 0.0540978, 0.0683364, // P Q R S
            0.0540687, 0.0673417, 0.0114135, 0.0304133  // T V W Y
    };

    const auto protein_num = std::unordered_map<char, int> {
            {'A', 0}, {'C', 1}, {'D', 2}, {'E', 3},
            {'F', 4}, {'G', 5}, {'H', 6}, {'I', 7},
            {'K', 8}, {'L', 9}, {'M', 10}, {'N', 11},
            {'P', 12}, {'Q', 13}, {'R', 14}, {'S', 15},
            {'T', 16}, {'V', 17}, {'W', 18}, {'Y', 19}
    };
}


MSV_HMM::MSV_HMM(const Profile_HMM &base_hmm)
    : model_length(base_hmm.model_length + 1)
{
    emission_scores.reserve(model_length);
    // base_hmm contains info about M[1]..M[base_hmm.model_length] in match_emissions.
    // Insert dummy node M[0]. It will be used for MSV algorithm initialization
    emission_scores.push_back({Log_scores_array<NUM_OF_AMINO_ACIDS>{}});

    std::transform(base_hmm.match_emissions.begin(), base_hmm.match_emissions.end(),
            std::back_inserter(emission_scores),
            [](const Probabilities_array<NUM_OF_AMINO_ACIDS>& node_match_emission) {
                auto log_scores = Log_scores_array<NUM_OF_AMINO_ACIDS>();
                for (size_t i = 0; i < NUM_OF_AMINO_ACIDS; ++i) {
                    log_scores[i] = std::log(node_match_emission[i] / background_frequencies[i]);
                }
                return log_scores;
            });

    // nu is expected number of hits (use 2.0 as a default).
    // https://github.com/EddyRivasLab/hmmer/blob/master/src/generic_msv.c#L39
    constexpr float nu = 2.0;

    tr_B_Mk = std::log2(2 / static_cast<float>(base_hmm.model_length * (base_hmm.model_length + 1)));
    tr_E_C  = std::log2((nu - 1.0f) / nu);
    tr_E_J  = std::log2(1.0f / nu);
}


void MSV_HMM::init_transitions_depend_on_seq(const Protein_sequence &seq) {
    tr_loop = seq.size() / static_cast<float>(seq.size() + 3);
    tr_move =          3 / static_cast<float>(seq.size() + 3);
}


Log_score MSV_HMM::run_on_sequence(Protein_sequence seq) {

    init_transitions_depend_on_seq(seq);

    // Insert dummy "residue" in seq
    seq = "#" + seq;

    // Dynamic programming matrix,
    // where L == seq.length(), k == model_length, both with dummies
    //
    //        M0 .. Mk-1 E J C N B
    // seq0
    // seq1
    // ..
    // seqL-1
    auto dp = std::vector<std::vector<Log_score>>(
        seq.length(), std::vector(model_length + 5, 0.0f));

    // E, J, C, N, B states indices
    const auto E = model_length;
    const auto J = model_length + 1;
    const auto C = model_length + 2;
    const auto N = model_length + 3;
    const auto B = model_length + 4;

    // dp matrix initialization

    for (auto& col : dp[0]) {
        col = minus_infinity;
    }
    dp[0][N] = 0.0;
    dp[0][B] = tr_move; // tr_N_B

    for (auto& row : dp) {
        row[0] = minus_infinity;
    }

    // MSV main loop
    for (size_t i = 1; i < seq.size(); ++i) {
        for (size_t j = 1; j < model_length; ++j) {
            dp[i][j] = emission_scores[j][protein_num.at(seq[i])] +
                       std::max(dp[i - 1][j - 1], dp[i - 1][B] + tr_B_Mk);
            dp[i][E] = std::max(dp[i][E], dp[i][j]);
        }

        dp[i][J] = std::max(dp[i - 1][J] + tr_loop, dp[i][E] + tr_E_J);
        dp[i][C] = std::max(dp[i - 1][C] + tr_loop, dp[i][E] + tr_E_C);
        dp[i][N] =          dp[i - 1][N] + tr_loop;
        dp[i][B] = std::max(dp[i    ][N] + tr_move, dp[i][J] + tr_move);
    }
    return dp.back()[C] + tr_move;
}


// SYCL implementation of MSV algorithm
// Baseline is run_on_sequence
// Memory optimization -- use 2-row dp matrix instead of seq.size-row matrix

// SYCL kernel names
class init_dp;
class init_N_B;
class M_states_handler;

float MSV_HMM::parallel_run_on_sequence(Protein_sequence seq) {
    init_transitions_depend_on_seq(seq);

    // Insert dummy "residue" in seq
    seq = "#" + seq;

    // Dynamic programming matrix,
    // where k == model_length with dummy
    //
    //   M0 .. Mk-1 E J C N B
    // 0
    // 1

    // E, J, C, N, B states indices
    const auto E = model_length;
    const auto J = model_length + 1;
    const auto C = model_length + 2;
    const auto N = model_length + 3;
    const auto B = model_length + 4;
    constexpr size_t rows = 2;
    const size_t cols = model_length + 5;


    namespace sycl = cl::sycl;
    {
        // Cannot capture 'this' in a SYCL kernel, introducing copy
        const auto move_score = tr_move;
        const auto B_Mk_score = tr_B_Mk;
        const auto num_of_M_states = model_length - 1;    // count without dummy M0

        // optional parameter for queue
        auto exception_handler = [] (const sycl::exception_list& exceptions) {
            for (const std::exception_ptr& e : exceptions) {
                try {
                    std::rethrow_exception(e);
                } catch(const sycl::exception& e) {
                    std::cout << "Caught asynchronous SYCL exception:\n"
                              << e.what() << '\n';
                }
            }
        };

        auto queue         = sycl::queue(sycl::default_selector(), exception_handler);
        auto dp            = sycl::buffer<float, 2>(sycl::range<2>(rows, cols));
        auto emissions_buf = sycl::buffer<float, 2>(emission_scores.data()->data(),
                sycl::range<2>(emission_scores.size(), NUM_OF_AMINO_ACIDS),
                sycl::property::buffer::use_host_ptr());

        // dp initialization, dp[1] left as is, i.e. "trash" values
        try {
            queue.submit([&] (sycl::handler& cgh) {
                auto dpA = dp.get_access<sycl::access::mode::discard_write, sycl::access::target::global_buffer>(cgh);
                cgh.parallel_for<init_dp>(sycl::range<1>(cols),
                        [=] (sycl::item<1> col_work_item) {
                            dpA[0][col_work_item.get_linear_id()] = minus_infinity;
                        });
            });

            queue.submit([&] (sycl::handler& cgh) {
                auto dpA = dp.get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(cgh);
                cgh.single_task<init_N_B>([=] () {
                    dpA[1][0] = minus_infinity;
                    dpA[0][N] = 0.0;
                    dpA[0][B] = move_score; // tr_N_B
                });
            });

            size_t cur_row = 1;
            size_t prev_row = 0;

            for (size_t i = 1; i < seq.size(); ++i) {
                auto protein_index = protein_num.at(seq[i]);
                queue.submit([&] (sycl::handler& cgh) {
                    auto dpA = dp.get_access<sycl::access::mode::write, sycl::access::target::global_buffer>(cgh);
                    auto emissions_bufA = emissions_buf.get_access<sycl::access::mode::read, sycl::access::target::constant_buffer>(cgh);

                    cgh.parallel_for<M_states_handler>(sycl::range<1>(num_of_M_states),
                        [=] (sycl::item<1> col_work_item) {
                            auto cur_col = col_work_item.get_linear_id() + 1;
                            dpA[cur_row][cur_col] = emissions_bufA[cur_col][protein_index]
                                + cl::sycl::fmax(dpA[prev_row][cur_col - 1], dpA[prev_row][B] + B_Mk_score);
                        });
                });
                prev_row = cur_row;
                cur_row = 1 - cur_row;
            }

            auto dpA_host = dp.get_access<sycl::access::mode::read>();
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    std::cout << dpA_host[i][j] << ' ';
                }
                std::cout << '\n';
            }
            queue.wait_and_throw();
        } catch (sycl::exception& e) {
            std::cout << e.what() << '\n';
        }
    }
    return 0;
}

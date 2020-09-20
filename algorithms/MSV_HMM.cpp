#include "MSV_HMM.hpp"
#include "MSV_kernel_store.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <unordered_map>

constexpr float minus_infinity = -std::numeric_limits<float>::infinity();

namespace {

// Default background frequencies for protein models.
// Numbers were taken from p7_AminoFrequencies hmmer function
constexpr auto background_frequencies = std::array<float, NUM_OF_AMINO_ACIDS>{
    0.0787945, 0.0151600, 0.0535222, 0.0668298, // A C D E
    0.0397062, 0.0695071, 0.0229198, 0.0590092, // F G H I
    0.0594422, 0.0963728, 0.0237718, 0.0414386, // K L M N
    0.0482904, 0.0395639, 0.0540978, 0.0683364, // P Q R S
    0.0540687, 0.0673417, 0.0114135, 0.0304133  // T V W Y
};

const auto amino_acid_num = std::unordered_map<char, int>{
    {'A', 0},  {'C', 1},  {'D', 2},  {'E', 3},  {'F', 4},  {'G', 5},  {'H', 6},  {'I', 7},  {'K', 8},  {'L', 9},
    {'M', 10}, {'N', 11}, {'P', 12}, {'Q', 13}, {'R', 14}, {'S', 15}, {'T', 16}, {'V', 17}, {'W', 18}, {'Y', 19}};
} // namespace

MSV_HMM::MSV_HMM(const Profile_HMM& base_hmm) : name(base_hmm.name), model_length(base_hmm.model_length) {
    // base_hmm contains info about M[0]..M[base_hmm.model_length] in match_emissions,
    // where node M[0] is zero-filled and will be used to simplify indexing.
    emission_scores = std::vector<Log_score>(NUM_OF_AMINO_ACIDS * model_length);

    for (size_t i = 0; i < model_length; ++i) {
        for (size_t j = 0; j < NUM_OF_AMINO_ACIDS; ++j) {
            const auto log_score = std::log(base_hmm.match_emissions[i][j] / background_frequencies[j]);
            emission_scores[j * model_length + i] = log_score;
        }
    }

    // nu is expected number of hits (use 2.0 as a default).
    // https://github.com/EddyRivasLab/hmmer/blob/master/src/generic_msv.c#L39
    constexpr float nu = 2.0;

    tr_B_Mk = std::log(2.0f / static_cast<float>(base_hmm.model_length * (base_hmm.model_length + 1)));
    tr_E_C = std::log((nu - 1.0f) / nu);
    tr_E_J = std::log(1.0f / nu);
}

void MSV_HMM::init_transitions_depend_on_seq(const Protein_sequence& seq) {
    // take into account # at the beginning of Protein_sequence
    auto size = seq.size() - 1;
    tr_loop = std::log(size / static_cast<float>(size + 3));
    tr_move = std::log(3 / static_cast<float>(size + 3));
}

Log_score MSV_HMM::run_on_sequence(const Protein_sequence& seq) {

    init_transitions_depend_on_seq(seq);

    // Dynamic programming matrix,
    // where L == seq.length(), k == model_length, both with dummies
    //
    //        M0 .. Mk-1 E J C N B
    // seq0
    // seq1
    // ..
    // seqL-1
    auto dp = std::vector<std::vector<Log_score>>(seq.length(), std::vector(model_length + 5, minus_infinity));

    // E, J, C, N, B states indices
    const auto E = model_length;
    const auto J = model_length + 1;
    const auto C = model_length + 2;
    const auto N = model_length + 3;
    const auto B = model_length + 4;

    // dp matrix initialization
    dp[0][N] = 0.0;
    dp[0][B] = tr_move; // tr_N_B

    // MSV main loop
    for (size_t i = 1; i < seq.size(); ++i) {
        const auto stride = amino_acid_num.at(seq[i]) * model_length;
        for (size_t j = 1; j < model_length; ++j) {
            dp[i][j] = emission_scores[stride + j] + std::max(dp[i - 1][j - 1], dp[i - 1][B] + tr_B_Mk);
            dp[i][E] = std::max(dp[i][E], dp[i][j]);
        }

        dp[i][J] = std::max(dp[i - 1][J] + tr_loop, dp[i][E] + tr_E_J);
        dp[i][C] = std::max(dp[i - 1][C] + tr_loop, dp[i][E] + tr_E_C);
        dp[i][N] = dp[i - 1][N] + tr_loop;
        dp[i][B] = std::max(dp[i][N] + tr_move, dp[i][J] + tr_move);
    }
    return dp.back()[C] + tr_move;
}

// SYCL implementation of MSV algorithm
// Baseline is run_on_sequence
// Memory optimization -- use 2-row dp matrix instead of seq.size-row matrix

// SYCL kernel names
class init_dp;
class init_N_B;
class reduction_step;
class E_J_C_N_B_states_handler;

Log_score MSV_HMM::parallel_run_on_sequence(const Protein_sequence& seq) {
    init_transitions_depend_on_seq(seq);

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
    using target = sycl::access::target;
    using mode = sycl::access::mode;
    {
        // Cannot capture 'this' in a SYCL kernel, introducing copy
        const auto move_score = tr_move;
        const auto loop_score = tr_loop;
        const auto E_J_score = tr_E_J;
        const auto E_C_score = tr_E_C;
        const auto num_of_real_M_states = model_length - 1; // count without dummy M0

        // optional parameter for queue
        auto exception_handler = [](const sycl::exception_list& exceptions) {
            for (const std::exception_ptr& e : exceptions) {
                try {
                    std::rethrow_exception(e);
                } catch (const sycl::exception& e) {
                    std::cout << "Caught asynchronous SYCL exception:\n" << e.what() << '\n';
                }
            }
        };

        auto queue = sycl::queue(sycl::default_selector(), exception_handler);

        auto dp_cur = sycl::buffer<float, 1>(sycl::range<1>(cols));
        auto dp_prev = sycl::buffer<float, 1>(sycl::range<1>(cols));
        // for swap purposes
        auto tmp = sycl::buffer<float, 1>();

        auto emissions_buf =
            sycl::buffer<float, 1>(emission_scores.data(), sycl::range<1>(NUM_OF_AMINO_ACIDS * model_length),
                                   sycl::property::buffer::use_host_ptr());

        // The algorithm that finds maximum requires data size to be even
        // M[0] is always minus_infinity, it does not affect the maximum of M states
        const auto should_use_M0 = static_cast<int>(num_of_real_M_states % 2 != 0);
        auto max_M_buf = sycl::buffer<float, 1>(sycl::range<1>(num_of_real_M_states + should_use_M0));
        const auto max_M_buf_size = max_M_buf.get_count();

        Spec_kernels_map kernels_map = get_spec_kernels_map();
        // TODO: assume calling only specialized version
        Kernels_pack spec_kernels;
        if (auto iter_to_pack = kernels_map.find(model_length);
            iter_to_pack != kernels_map.end())
        {
            spec_kernels = iter_to_pack->second;
        } else {
            std::cout << "Failure! Can not find specialized kernel with length " << model_length << "!\n";
            return -1;
        }

        // dp initialization, dp[1] left as is, i.e. "trash" values
        try {
            queue.submit([&](sycl::handler& cgh) {
                auto dp_prev_A = dp_prev.get_access<mode::discard_write, target::global_buffer>(cgh);
                cgh.parallel_for<init_dp>(sycl::range<1>(cols), [=](sycl::item<1> col_work_item) {
                    dp_prev_A[col_work_item.get_linear_id()] = minus_infinity;
                });
            });

            queue.submit([&](sycl::handler& cgh) {
                auto dp_cur_A = dp_cur.get_access<mode::write, target::global_buffer>(cgh);
                auto dp_prev_A = dp_prev.get_access<mode::write, target::global_buffer>(cgh);
                cgh.single_task<init_N_B>([=]() {
                    dp_cur_A[0] = minus_infinity;
                    dp_prev_A[N] = 0.0;
                    dp_prev_A[B] = move_score; // tr_N_B
                });
            });

            for (size_t i = 1; i < seq.size(); ++i) {
                const auto stride = amino_acid_num.at(seq[i]) * NUM_OF_AMINO_ACIDS;

                // Calculate M states
                auto m_handler_gen = std::get<M_states_handler_gen_num>(spec_kernels);
                queue.submit(m_handler_gen(dp_cur, dp_prev, emissions_buf, amino_acid_num, seq, i));

                // Data preparation to get max from M states
                auto m_copy_gen = std::get<M_copy_gen_num>(spec_kernels);
                queue.submit(m_copy_gen(dp_cur, max_M_buf));

                // Find max from M states, max_M_buf size is even
                // Can it be improved with local memory usage?
                auto left_half_size = max_M_buf_size / 2;

                // Rest of the reduction without last comparison of bufA[0] and bufA[1]
                while (left_half_size > 1) {
                    queue.submit([&](sycl::handler& cgh) {
                        auto bufA = max_M_buf.get_access<mode::read_write, target::global_buffer>(cgh);

                        cgh.parallel_for<reduction_step>(sycl::range<1>(left_half_size), [=](sycl::item<1> item) {
                            auto id = item.get_linear_id();
                            auto offset = left_half_size + id;
                            bufA[id] = sycl::fmax(bufA[id], bufA[offset]);
                        });
                    });

                    left_half_size += (left_half_size % 2);
                    left_half_size /= 2;
                }

                queue.submit([&](sycl::handler& cgh) {
                    auto dp_cur_A = dp_cur.get_access<mode::write, target::global_buffer>(cgh);
                    auto dp_prev_A = dp_cur.get_access<mode::read, target::global_buffer>(cgh);
                    auto bufA = max_M_buf.get_access<mode::read, target::global_buffer>(cgh);

                    cgh.single_task<E_J_C_N_B_states_handler>([=]() {
                        dp_cur_A[E] = sycl::fmax(bufA[0], bufA[1]);
                        dp_cur_A[J] = sycl::fmax(dp_prev_A[J] + loop_score, dp_cur_A[E] + E_J_score);
                        dp_cur_A[C] = sycl::fmax(dp_prev_A[C] + loop_score, dp_cur_A[E] + E_C_score);
                        dp_cur_A[N] = dp_prev_A[N] + loop_score;
                        dp_cur_A[B] = sycl::fmax(dp_cur_A[N] + move_score, dp_cur_A[J] + move_score);
                    });
                });

                tmp = std::move(dp_cur);
                dp_cur = std::move(dp_prev);
                dp_prev = std::move(tmp);
            }

            auto dpA_host = dp_prev.get_access<mode::read>();
            return dpA_host[C] + move_score;
        } catch (sycl::exception& e) {
            std::cout << e.what() << '\n';
        }
    }
    return 0;
}

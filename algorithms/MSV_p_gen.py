import math
import sys
import os
import fileinput


def neg_ln_to_prob(prob_inside_ln: float):
    return math.exp(-1 * prob_inside_ln)


model_length: int = 0
NUM_OF_AMINO_ACIDS: int = 20
nu: float = 2.0

cpp_MSV_initial = f'''#include \"MSV_HMM_spec.hpp\"
    
#include <CL/sycl.hpp>
#include <iostream>
#include <unordered_map>

namespace {{
const auto amino_acid_num = std::unordered_map<char, int>{{
    {{'A', 0}},  {{'C', 1}},  {{'D', 2}},  {{'E', 3}},  {{'F', 4}},  {{'G', 5}},  {{'H', 6}},  {{'I', 7}},  {{'K', 8}},  {{'L', 9}},
    {{'M', 10}}, {{'N', 11}}, {{'P', 12}}, {{'Q', 13}}, {{'R', 14}}, {{'S', 15}}, {{'T', 16}}, {{'V', 17}}, {{'W', 18}}, {{'Y', 19}}}};
}}

'''

background_frequencies: list = [
    0.0787945, 0.0151600, 0.0535222, 0.0668298,  # A C D E
    0.0397062, 0.0695071, 0.0229198, 0.0590092,  # F G H I
    0.0594422, 0.0963728, 0.0237718, 0.0414386,  # K L M N
    0.0482904, 0.0395639, 0.0540978, 0.0683364,  # P Q R S
    0.0540687, 0.0673417, 0.0114135, 0.0304133   # T V W Y
]

# generate specialized code
for hmm_file in sys.argv[1:]:
    hmm_name: str = hmm_file[:-4]
    slash_index: int = hmm_name.rfind('/')
    if slash_index != -1:
        hmm_name = hmm_name[slash_index+1:]

    # emulate M0
    match_emissions: list = [[0] * NUM_OF_AMINO_ACIDS]

    # read match_emissions
    with open(hmm_file, 'r') as hmm:
        while True:
            line: str = hmm.readline().lstrip()
            if line.startswith('COMPO'):
                break
            if line.startswith('LENG'):
                model_length = int(line.split()[1])

        hmm.readline()
        hmm.readline()
        line: str = hmm.readline().strip()
        while line != '//':
            data: list = line.split()[1:(NUM_OF_AMINO_ACIDS + 1)]
            match_emissions.append(list(map(lambda d: neg_ln_to_prob(float(d)), data)))
            hmm.readline()
            hmm.readline()
            line = hmm.readline().strip()

    # take into account dummy node M0
    model_length += 1

    emissions_scores: list = [0] * NUM_OF_AMINO_ACIDS * model_length

    for i in range(model_length):
        for j in range(NUM_OF_AMINO_ACIDS):
            if match_emissions[i][j] == 0:
                emissions_scores[j * model_length + i] = -math.inf
            else:
                log_score: float = math.log(match_emissions[i][j] / background_frequencies[j])
                emissions_scores[j * model_length + i] = log_score

    # embed emissions_scores
    cpp_emissions_scores_initializer: str = '{\n'
    for i in range(model_length):
        cpp_emissions_scores_initializer += '      '
        stride: int = i * NUM_OF_AMINO_ACIDS
        cpp_emissions_scores_initializer += ', '.join(map(lambda f: '%.6f' % f, emissions_scores[stride:stride+20])) + ',\n'
    cpp_emissions_scores_initializer += '    }'

    # constants calculated from model_length
    B_Mk_score: str = 'static_cast<float>(' + str(math.log(2.0 / (model_length * (model_length + 1)))) + ')'
    E_C_score: str = 'static_cast<float>(' + str(math.log((nu - 1.0) / nu)) + ')'
    E_J_score: str = 'static_cast<float>(' + str(math.log(1.0 / nu)) + ')'

    E: int = model_length
    J: int = model_length + 1
    C: int = model_length + 2
    N: int = model_length + 3
    B: int = model_length + 4
    rows: int = 2
    cols: int = model_length + 5

    num_of_real_M_states: int = model_length - 1
    should_use_M0: int = num_of_real_M_states % 2 != 0
    max_M_buf_size: int = num_of_real_M_states + should_use_M0

    cpp_newline = "\'\\n\'"
    init_dp: str = f'init_dp_spec_{hmm_name}'
    init_N_B: str = f'init_N_B_spec_{hmm_name}'
    M_states_handler: str = f'M_states_handler_spec_{hmm_name}'
    copy_M: str = f'copy_M_spec_{hmm_name}'
    reduction_step: str = f'reduction_step_spec_{hmm_name}'
    E_J_C_N_B_states_handler: str = f'E_J_C_N_B_states_handler_spec_{hmm_name}'

    # TODO: unroll while

    p_gen = f'''// MSV specialized at HMM from {hmm_file}
// specialized SYCL kernel names
class {init_dp};
class {init_N_B};
class {M_states_handler};
class {copy_M};
class {reduction_step};
class {E_J_C_N_B_states_handler};

Log_score MSV_HMM_spec::parallel_run_on_sequence_{hmm_name}(const Protein_sequence& seq) {{
    // workaround to make C++ understand Python output
    constexpr float inf = std::numeric_limits<float>::infinity();

    // init scores that dependens on seq
    const auto size = seq.size() - 1;
    const auto loop_score = std::log(size / static_cast<float>(size + 3));
    const auto move_score = std::log(3 / static_cast<float>(size + 3));
    static constexpr auto emission_scores = std::array<Log_score, {NUM_OF_AMINO_ACIDS * model_length}>{cpp_emissions_scores_initializer};

    namespace sycl = cl::sycl;
    using target = sycl::access::target;
    using mode = sycl::access::mode;
    {{
        auto exception_handler = [](const sycl::exception_list& exceptions) {{
            for (const std::exception_ptr& e : exceptions) {{
                try {{
                    std::rethrow_exception(e);
                }} catch (const sycl::exception& e) {{
                    std::cout << "Caught asynchronous SYCL exception:" << {cpp_newline} << e.what() << {cpp_newline};
                }}
            }}
        }};

        auto queue = sycl::queue(sycl::default_selector(), exception_handler);
        auto dp = sycl::buffer<float, 2>(sycl::range<2>({rows}, {cols}));
        auto emissions_buf =
            sycl::buffer<float, 1>(emission_scores.data(), sycl::range<1>({NUM_OF_AMINO_ACIDS * model_length}),
                                   sycl::property::buffer::use_host_ptr());

        auto max_M_buf = sycl::buffer<float, 1>(sycl::range<1>({num_of_real_M_states + should_use_M0}));

        try {{
            // dp initialization
            queue.submit([&](sycl::handler& cgh) {{
                auto dpA = dp.get_access<mode::discard_write, target::global_buffer>(cgh);
                cgh.parallel_for<{init_dp}>(sycl::range<1>({cols}), [=](sycl::item<1> col_work_item) {{
                    dpA[0][col_work_item.get_linear_id()] = -inf;
                }});
            }});

            queue.submit([&](sycl::handler& cgh) {{
                auto dpA = dp.get_access<mode::write, target::global_buffer>(cgh);
                cgh.single_task<{init_N_B}>([=]() {{
                    dpA[1][0] = -inf;
                    dpA[0][{N}] = 0.0;
                    dpA[0][{B}] = move_score; // tr_N_B
                }});
            }});

            size_t cur_row = 1;
            size_t prev_row = 0;

            for (size_t i = 1; i < seq.size(); ++i) {{
                const auto stride = amino_acid_num.at(seq[i]) * {NUM_OF_AMINO_ACIDS};

                // Calculate M states
                queue.submit([&](sycl::handler& cgh) {{
                    auto dpA = dp.get_access<mode::write, target::global_buffer>(cgh);
                    auto emissions_bufA = emissions_buf.get_access<mode::read, target::constant_buffer>(cgh);

                    cgh.parallel_for<{M_states_handler}>(
                        sycl::range<1>({num_of_real_M_states}), [=](sycl::item<1> col_work_item) {{
                            auto cur_col = col_work_item.get_linear_id() + 1;
                            dpA[cur_row][cur_col] =
                                emissions_bufA[stride + cur_col] +
                                sycl::fmax(dpA[prev_row][cur_col - 1], dpA[prev_row][{B}] + {B_Mk_score});
                        }});
                }});

                // Data preparation to get max from M states
                queue.submit([&](sycl::handler& cgh) {{
                    auto dpA = dp.get_access<mode::read, target::global_buffer>(cgh);
                    auto bufA = max_M_buf.get_access<mode::discard_write, target::global_buffer>(cgh);

                    cgh.parallel_for<{copy_M}>(sycl::range<1>({max_M_buf_size}), [=](sycl::item<1> col_work_item) {{
                        auto id = col_work_item.get_linear_id();
                        bufA[id] = dpA[cur_row][id {'+ 1' if should_use_M0 == 0 else ''}];
                    }});
                }});

                auto left_half_size = {int(max_M_buf_size / 2)};

                while (left_half_size > 1) {{
                    queue.submit([&](sycl::handler& cgh) {{
                        auto bufA = max_M_buf.get_access<mode::read_write, target::global_buffer>(cgh);

                        cgh.parallel_for<{reduction_step}>(sycl::range<1>(left_half_size), [=](sycl::item<1> item) {{
                            auto id = item.get_linear_id();
                            auto offset = left_half_size + id;
                            bufA[id] = sycl::fmax(bufA[id], bufA[offset]);
                        }});
                    }});

                    left_half_size += (left_half_size % 2);
                    left_half_size /= 2;
                }}

                queue.submit([&](sycl::handler& cgh) {{
                    auto dpA = dp.get_access<mode::read_write, target::global_buffer>(cgh);
                    auto bufA = max_M_buf.get_access<mode::read, target::global_buffer>(cgh);
                    cgh.single_task<{E_J_C_N_B_states_handler}>([=]() {{
                        dpA[cur_row][{E}] = sycl::fmax(bufA[0], bufA[1]);
                        dpA[cur_row][{J}] = sycl::fmax(dpA[prev_row][{J}] + loop_score, dpA[cur_row][{E}] + {E_J_score});
                        dpA[cur_row][{C}] = sycl::fmax(dpA[prev_row][{C}] + loop_score, dpA[cur_row][{E}] + {E_C_score});
                        dpA[cur_row][{N}] = dpA[prev_row][{N}] + loop_score;
                        dpA[cur_row][{B}] = sycl::fmax(dpA[cur_row][{N}] + move_score, dpA[cur_row][{J}] + move_score);
                    }});
                }});

                prev_row = cur_row;
                cur_row = 1 - cur_row;
            }}

            auto dpA_host = dp.get_access<mode::read>();
            return dpA_host[prev_row][{C}] + move_score;
        }} catch (sycl::exception& e) {{
            std::cout << e.what() << {cpp_newline};
        }}
    }}
    return 0;
}}\n\n'''

    # Write out result of p_gen
    out_file_name = 'MSV_HMM_spec'
    out_file_name_cpp = out_file_name + '.cpp'
    out_file_name_header = out_file_name + '.hpp'

    if os.path.exists(out_file_name_cpp) and os.stat(out_file_name_cpp).st_size > 0:
        with open(out_file_name_cpp, 'a') as out:
            out.write(p_gen)
    else:
        with open(out_file_name_cpp, 'w') as out:
            out.write(cpp_MSV_initial)
            out.write(p_gen)

    cpp_MSV_header = f'''#pragma once
    
#include "FASTA_protein_sequences.hpp"

using Log_score = float;

class MSV_HMM_spec {{
  public:
    static Log_score parallel_run_on_sequence_{hmm_name}(const Protein_sequence& seq);
}};
'''

    # May be there is more elegant way to replace last string in file
    if os.path.exists(out_file_name_header) and os.stat(out_file_name_header).st_size > 0:
        with fileinput.input(files=out_file_name_header, inplace=True) as f:
            for line in f:
                if '};' in line:
                    line = f'    static Log_score parallel_run_on_sequence_{hmm_name}(const Protein_sequence& seq);\n' + line
                sys.stdout.write(line)
    else:
        with open(out_file_name_header, 'w') as out:
            out.write(cpp_MSV_header)

import math
import sys
import os
import fileinput

nu: float = 2.0
NUM_OF_AMINO_ACIDS: int = 20
lengths: list = []

# read profile HMMs' names and lengths
for hmm_file in sys.argv[1:]:
    with open(hmm_file, 'r') as hmm:
        name: str = ""
        while True:
            line: str = hmm.readline().lstrip()
            if line.startswith('LENG'):
                # take into account dummy node M0
                lengths.append(int(line.split()[1]) + 1)
                break

output_file: str = 'MSV_kernel_store.hpp'

lengths.sort()

with fileinput.input(files=output_file, inplace=True) as f:
        for line in f:
            if 'using Spec_kernels_map = std::map<HMM_size, Kernels_pack>;' in line:
                line += '\n'
                for length in lengths:
                    # constants calculated from length
                    B_Mk_score: str = 'static_cast<float>(' + str(math.log(2.0 / (length * (length + 1)))) + ')'
                    B: int = length + 4

                    num_of_real_M_states: int = length - 1
                    should_use_M0: int = num_of_real_M_states % 2 != 0
                    max_M_buf_size: int = num_of_real_M_states + should_use_M0

                    m_states = f'''M_states_handler_kernel_{length}'''
                    m_copy = f'''M_copy_kernel_{length}'''
                    line += f'''class {m_states};\n'''
                    line += f'''class {m_copy};\n\n'''
                    # Kernel generation
                    line += \
f'''Command_group_function M_states_handler_{length}(
    Dp_cur_buffer& dp_cur, Dp_prev_buffer& dp_prev, Em_buffer& emissions_buf,
    const std::unordered_map<char, int>& amino_acid_num, const std::string& seq, const size_t& i) 
{{
    return [&](sycl::handler& cgh) {{
                auto dp_cur_A = dp_cur.get_access<mode::write, target::global_buffer>(cgh);
                auto dp_prev_A = dp_prev.get_access<mode::read, target::global_buffer>(cgh);
                auto emissions_bufA = sycl::buffer<float, 1>(
                                    emissions_buf,
                                    sycl::id<1>(amino_acid_num.at(seq[i]) * {length}),
                                    sycl::range<1>({length}))
                                .get_access<mode::read, target::constant_buffer>(cgh);

                cgh.parallel_for<{m_states}>(
                    sycl::range<1>({num_of_real_M_states}), [=](sycl::item<1> col_work_item) {{
                        auto cur_col = col_work_item.get_linear_id() + 1;
                        dp_cur_A[cur_col] =
                            emissions_bufA[cur_col] +
                            sycl::fmax(dp_prev_A[cur_col - 1], dp_prev_A[{B}] + {B_Mk_score});
                    }});
            }};
}}\n\n'''

                    line += \
f'''Command_group_function M_copy_{length}(Dp_cur_buffer& dp_cur, M_buffer& max_M_buf) {{
    return [&](sycl::handler& cgh) {{
                auto dp_cur_A = dp_cur.get_access<mode::read, target::global_buffer>(cgh);
                auto bufA = max_M_buf.get_access<mode::discard_write, target::global_buffer>(cgh);

                cgh.parallel_for<M_copy_kernel_{length}>(sycl::range<1>({max_M_buf_size}), [=](sycl::item<1> col_work_item) {{
                    auto id = col_work_item.get_linear_id();
                    bufA[id] = dp_cur_A[id + {1 - should_use_M0}];
                }});
    }};
}}\n\n'''
            elif 'return res_map;' in line:
                insert_data_line: str = ''
                for length in lengths:
                    insert_data_line += f'''    res_map[{length}] = std::make_pair(&M_states_handler_{length}, M_copy_{length});\n'''
                line = insert_data_line + line
            sys.stdout.write(line)

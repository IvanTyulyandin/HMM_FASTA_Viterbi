with open('MSV_kernel_store.hpp', 'w') as file:
    file.write('''#include <CL/sycl.hpp>
#include <unordered_map>

namespace sycl = cl::sycl;
using target = sycl::access::target;
using mode = sycl::access::mode;

using Dp_cur_buffer = sycl::buffer<float, 1>;
using Dp_prev_buffer = sycl::buffer<float, 1>;
using Em_buffer = sycl::buffer<float, 1>;
using M_buffer = sycl::buffer<float, 1>;

using Command_group_function = std::function<void(sycl::handler&)>;
// pass by ref to save in stateful lambda
using M_states_handler_gen = Command_group_function(*)(
    Dp_cur_buffer&, Dp_prev_buffer&, Em_buffer&, const std::unordered_map<char, int>&, const std::string&, const size_t&);
using M_copy_gen = Command_group_function(*)(Dp_cur_buffer&, M_buffer&);

using HMM_size = size_t;
using Kernels_pack = std::pair<M_states_handler_gen, M_copy_gen>;
constexpr size_t M_states_handler_gen_num = 0;
constexpr size_t M_copy_gen_num = 1;

using Spec_kernels_map = std::map<HMM_size, Kernels_pack>;


Spec_kernels_map get_spec_kernels_map() {
    auto res_map = Spec_kernels_map();
    return res_map;
}''')

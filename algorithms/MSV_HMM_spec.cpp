#include "MSV_HMM_spec.hpp"

#include <CL/sycl.hpp>
#include <iostream>
#include <unordered_map>

namespace {
const auto amino_acid_num = std::unordered_map<char, int>{
    {'A', 0},  {'C', 1},  {'D', 2},  {'E', 3},  {'F', 4},  {'G', 5},  {'H', 6},  {'I', 7},  {'K', 8},  {'L', 9},
    {'M', 10}, {'N', 11}, {'P', 12}, {'Q', 13}, {'R', 14}, {'S', 15}, {'T', 16}, {'V', 17}, {'W', 18}, {'Y', 19}};
}

// MSV specialized at HMM from 100.hmm
// specialized SYCL kernel names
class init_dp_spec_100;
class init_N_B_spec_100;
class M_states_handler_spec_100;
class copy_M_spec_100;
class reduction_step_spec_100;
class E_J_C_N_B_states_handler_spec_100;

Log_score MSV_HMM_spec::parallel_run_on_sequence_100(const Protein_sequence& seq) {
    // workaround to make C++ understand Python output
    constexpr float inf = std::numeric_limits<float>::infinity();

    // init scores that dependens on seq
    const auto size = seq.size() - 1;
    const auto loop_score = std::log(size / static_cast<float>(size + 3));
    const auto move_score = std::log(3 / static_cast<float>(size + 3));
    auto emission_scores = std::array<Log_score, 2020>{
        -inf, -0.121198, -0.274148, -0.243048, -0.122408, -0.162818, -0.092048, -0.614358, 0.073802, 0.086662, 0.039392, 0.206482, -0.484278, -0.600998, -0.785958, -0.187808, 0.289802, 1.848122, -0.469828, -0.484528,
        -0.383848, -0.777078, -0.770518, -0.579048, -0.009368, 0.163292, -0.114658, 0.366712, 0.072602, -0.484768, 1.632482, 0.899532, -0.049938, -0.885358, 0.204332, 0.303262, 0.104272, -0.816608, 0.065502, 0.644182,
        -0.890058, 0.249792, -0.134348, -0.421198, -0.286828, -0.603028, 0.643502, -0.134618, -0.047108, 0.028582, 0.475912, 0.015552, 0.127992, 0.122012, -0.030368, 0.161722, 0.057972, -0.033318, -0.843498, -0.067408,
        0.295452, 0.194602, 0.334812, 0.472992, 0.129462, 0.188652, 0.447692, 0.156952, -0.939928, 0.121842, 1.703462, -0.508228, -0.019788, -0.246108, -0.070838, -0.608358, -0.043148, -0.260768, 0.129162, -0.706038,
        -0.095488, -1.697398, -1.091368, -0.721398, -0.212138, -0.081618, -1.461108, 0.098422, 0.248742, -1.196058, -0.406308, -0.299948, -0.117768, 0.203652, -0.032608, 0.010642, 0.061992, 0.708252, -0.175908, -0.239188,
        -0.498218, -inf, 0.080805, -0.875315, -0.625755, -0.676715, -0.753235, -1.008855, -0.998195, -0.842875, -0.336745, -1.154265, -0.078235, -1.195555, -0.292785, -0.893855, -0.727895, 0.942455, -0.155735, -0.315385,
        -0.577365, -0.649105, -0.527925, -0.430685, -0.906685, -0.879735, -0.238385, -0.972405, -0.751005, 0.082885, -1.281235, -0.038255, -0.736435, -0.250495, -0.626235, -0.461345, -0.671065, 1.918605, -0.466285, -0.972955,
        -0.757945, -0.553995, 2.015105, -0.511235, -1.359175, -1.136605, -0.284495, -0.442135, 1.006485, 0.051765, -0.103665, -0.107015, -0.895795, -0.717615, 0.073175, -1.114585, -0.856455, -0.766835, -0.684165, -0.624115,
        0.356735, -0.117655, -0.790915, -0.851715, -0.640725, -0.642245, -0.615525, -0.521715, -0.756265, -0.677095, -0.117185, -0.031275, -1.066865, -0.767745, -0.000815, -0.804635, -1.001175, -0.522925, -0.096325, 0.077135,
        -1.414195, -0.823285, -1.230765, -1.195015, -0.395195, -0.401465, -0.346455, -1.106245, -0.196015, -0.308585, -0.883865, -0.939925, -0.853465, -0.847715, -0.276785, -0.606315, -0.708395, -0.005765, -0.008675, -0.553895,
        -1.028915, -0.411195, -inf, -1.251801, -0.256411, -0.338031, -0.091171, -0.088801, 1.027659, -0.450491, 0.164689, 0.198859, 1.180409, -0.575031, 2.063059, -1.877981, -0.496661, -0.175331, -0.880401, -0.822441,
        -1.361571, -0.824651, -0.530921, -1.807941, -2.035251, -0.178911, -0.029611, -0.484401, 0.345569, 0.090749, -0.219541, 1.993389, -0.765491, 0.506239, -0.175941, -1.750831, -0.178381, 0.052749, -0.427881, -2.071851,
        0.254689, 0.192319, -2.075451, -0.549591, 1.061299, 1.482199, 1.366359, -1.899181, -0.171651, -1.438291, -0.061621, -0.552491, -0.655661, 0.308479, 0.128119, -0.417761, 1.566799, 0.449599, 0.215009, -0.139641,
        -1.462881, -0.179991, -0.551591, 0.166749, 0.504569, 0.039409, -0.019291, 0.189909, -0.125361, 0.127529, -1.705381, 0.095129, -0.770161, -0.644021, 0.055919, -1.523281, 0.002139, -0.501951, -0.152581, -1.205321,
        -0.155591, 2.238109, 0.566729, -1.957281, -1.251201, -2.116171, -1.017781, -0.777861, -1.647021, -0.746401, -0.720591, -2.024181, -0.133371, -0.159491, -0.088711, -0.491361, -0.157741, -0.090171, -0.931371, -0.948391,
        -0.289891, 1.210929, -1.909341, -inf, -0.894664, 0.131486, -0.128544, 0.180656, 0.200536, 0.501986, -0.203154, 0.525486, -0.070594, 0.957846, -0.484674, 0.421966, -1.527294, -0.512824, 0.170146, -0.742404,
        -0.857794, -1.056484, -0.983434, -0.178154, -1.558234, -1.710214, -0.168704, 0.276606, -0.606594, 1.171736, 0.370916, 0.117026, 0.530566, -0.689014, 0.413496, -0.021864, -1.566994, 0.210616, 0.406226, 0.005176,
        -1.764924, 0.984526, 0.611886, -1.729234, -0.172134, 0.388926, 1.053476, 0.511296, -1.624944, 0.314496, -1.067264, 0.435216, -0.072364, -0.315634, 0.585236, 0.579826, -0.077214, 0.613156, 0.467286, 0.408096,
        0.160756, -1.273954, 0.243606, -0.211864, 0.391596, 0.502936, 0.414706, 0.431056, 0.352106, 0.262726, 0.830556, -1.577424, 0.414036, -0.752484, -0.171514, 0.465386, -1.154654, 0.426866, 0.055006, 0.222876,
        -0.855974, 0.182006, 0.252676, 0.346336, -2.019774, -0.914834, -1.893954, -1.048034, -0.866314, -1.708534, -0.760074, -0.856004, -1.913064, 0.016056, 0.108446, 0.214096, -0.373134, 0.159466, 0.208506, -0.576034,
        -0.593454, 0.039336, 0.802306, -1.553614, -inf, 0.545448, -1.191012, -1.196062, -1.070442, -1.104222, -1.289612, -1.398342, -1.114002, -1.293592, -1.429432, -1.046952, -1.541482, 0.636168, -0.276852, -0.952242,
        -1.055542, -1.108102, -0.011252, -1.584422, -0.071072, 0.042308, -0.027902, -1.071942, -1.175272, -1.356992, -1.239462, -1.018592, -0.575452, -1.605692, -0.975222, -1.024382, -0.058732, 0.137828, -0.725142, -0.897382,
        -0.405942, -0.012792, -1.252282, -1.012302, 0.506918, -0.398902, -1.173282, -1.626572, -1.411922, -0.369512, -0.615802, 0.121708, -0.848702, -0.202262, -0.235402, -1.169582, -0.951462, -0.333482, -1.423382, -1.122812,
        -1.014902, -0.984432, 0.715738, -0.648952, -0.240772, -1.050582, -1.120682, -0.851512, -0.830432, -0.818392, -0.716912, -1.002992, -0.039062, -0.973802, -1.062412, -1.505152, -1.015492, 0.985998, -1.066012, -1.396442,
        -0.702752, -0.077072, -0.678602, -1.413422, -0.361152, -0.088622, -1.639752, -0.600822, -1.258312, -1.375592, 0.747848, -1.386682, -1.593202, -0.166352, -1.307272, -0.923362, -1.127262, -1.084562, -0.595382, -1.004342,
        0.207928, 0.015278, -0.498362, -1.303172, 0.020988, -inf, -1.126784, -0.885024, 0.859506, -0.591724, 0.296816, -0.692144, -1.004944, -0.725134, 1.612616, -0.652974, -0.399354, -0.614674, -1.722664, -1.032574,
        -0.841094, -0.071354, -0.494234, -1.416724, 2.247556, -1.030564, -1.763384, -1.953614, -0.893164, -0.503714, 1.913466, -0.708264, -0.736504, 0.060356, -0.614754, -0.418404, -0.690034, 0.463016, -1.746944, -0.807754,
        -0.708744, -0.825874, -1.995964, -0.636994, -0.596874, -2.016584, -0.731244, 0.082826, 0.112286, 0.086756, -1.795154, -0.797354, -1.160304, -0.760374, -0.900374, -0.936214, 0.024176, -0.739664, -0.158024, -0.441314,
        -0.462594, -0.687324, -0.793484, -1.464724, -0.801224, -0.777244, -0.724404, -0.717774, -0.755964, -0.755634, -0.760504, -0.755734, -0.602674, -1.664084, -0.737404, -0.385924, -1.088444, -0.746214, -1.269924, -0.665874,
        -1.037874, -0.757834, -1.256754, -0.083544, -0.818984, -0.681524, -1.658084, -1.326234, -1.915204, -0.739894, -0.594544, -1.744644, -0.474304, 2.033106, -1.914484, -0.895114, -0.906894, -0.512044, 1.545426, -0.577034,
        0.040756, -0.771424, -0.832414, -0.902484, -0.759884, -1.768074, -inf, -0.321426, 0.157054, -0.108216, 0.087404, 0.117174, 0.122394, -0.039316, 0.181154, -0.387106, 0.058394, -0.470026, -0.194066, -0.922646,
        3.084134, 0.696154, -0.594646, -0.775636, -0.678776, -1.009546, 0.323734, -1.083106, -1.221246, -0.256056, 0.289054, -0.713846, 0.104614, 0.108624, 0.077504, -0.135726, -0.595576, 0.061054, 0.023474, -1.021276,
        0.213934, 0.150784, 0.125584, -1.251386, 0.137344, 0.127844, -1.126426, -0.128816, 0.333464, -0.025606, 0.365244, -1.260866, 0.301194, -0.365586, 0.288084, -0.045986, -0.141866, 0.154014, 0.168504, -0.005836,
        0.010804, 0.162824, 0.492824, 0.098744, -0.103386, 0.187864, -0.069866, 0.162304, 0.176834, 0.147234, 0.307594, 0.700364, 0.106364, 0.167544, -1.077326, 0.207944, -0.665256, 0.109604, 0.452814, -0.430226,
        0.170824, -0.006436, 0.107794, -0.465546, 0.103444, -0.318976, 0.503124, -0.833646, -0.514836, -1.547546, -0.946326, -0.876006, -0.171956, -0.764786, -0.940216, -1.355636, -0.064706, 0.046484, 1.637894, -0.068306,
        1.234124, 0.757764, -0.206696, -0.220236, 0.008094, 0.046194, -0.951996, -inf, 0.847332, -0.982778, -1.044288, -0.885888, -0.918418, -1.156618, -1.215088, -0.967178, -1.150508, -1.320308, -0.791808, -1.514528,
        0.712592, -1.241128, -0.767478, -0.854748, -0.670688, 0.622572, -1.651988, -0.634308, 0.597422, 1.045152, -1.148348, -1.005398, -1.213728, -1.089768, -0.841548, -0.357578, -1.474168, -0.531198, -0.849338, -0.140068,
        0.453152, -0.517038, -0.719408, -0.170518, 1.379922, -1.113758, -0.845618, 0.634712, -0.152418, -1.032568, -1.543548, -1.300418, 1.860612, -0.399728, 0.641492, -0.574678, 0.302542, 0.336592, -1.030268, -0.622578,
        0.113072, -1.302558, -0.978028, -0.805648, -0.788348, -0.287008, -0.379088, 0.236732, -0.894208, -0.977418, -0.629628, -0.635158, -0.487948, -0.514328, -0.841128, 0.337112, -0.806678, -0.642998, -1.228768, -0.810898,
        0.635962, -0.906688, -1.182988, -0.498898, 0.882542, -0.408208, -1.601988, -0.914308, -1.488798, -1.636038, 1.066252, -0.847258, -1.374658, -1.127048, -1.197178, -1.429008, 0.186722, -1.178798, -0.913458, -0.958078,
        -0.902028, -0.629558, -0.830158, 0.843482, 0.212812, -0.456448, -1.162258, 0.782012, -inf, -0.651859, 0.944901, 0.378091, 0.475411, 0.786431, 0.933511, 1.979031, 0.895511, -0.346069, 0.217831, -0.372899,
        -0.242289, -1.285829, -0.233179, 0.601421, -0.572239, -0.764279, -0.764629, -1.082819, 0.437691, -1.223419, -1.458099, 0.144201, 1.152261, -0.699159, 0.402501, 0.612331, 0.430501, -0.153019, -0.547019, 0.259831,
        0.127311, -1.227609, 0.457001, 0.415351, 0.128131, -1.482579, 0.567001, 0.378981, -1.482379, -0.074259, 0.341191, 0.029331, 0.197381, -1.370429, 0.283911, -0.790699, 0.415641, -0.035369, -0.139659, 0.418711,
        0.440871, 0.075841, 0.132551, 0.458781, 0.460081, 0.497611, -0.799469, 0.344501, -0.046869, 0.438191, 0.460301, 0.450411, 0.429071, 0.413421, 0.340361, 0.443411, -1.225749, 0.426441, -0.633679, 1.195011,
        0.519071, -0.884609, 0.578471, 0.720591, 0.403131, -0.639569, 0.361191, -0.367679, 0.407451, -1.637819, 0.233911, -1.647269, -0.884379, -0.858209, -1.439459, -0.701419, -0.961709, -1.562689, 0.429961, 0.436601,
        0.441221, -0.311789, 0.287891, 0.345091, -0.373099, -0.389949, 0.252241, 0.276061, -1.271939, -inf, 0.156931, -0.767529, -1.096059, -0.936699, -0.951909, -1.150559, -1.192829, -0.980469, -1.307089, -1.306169,
        -1.023799, -1.591419, 1.115121, -1.188559, -0.183599, -1.067359, -1.029909, 1.347371, -1.760769, -0.682739, 1.566191, 1.374281, -1.113849, -0.946319, -1.418389, -1.106919, -0.903989, -0.435809, -1.549999, -0.875089,
        -0.930769, -0.154969, 1.670781, -0.504779, -0.764309, -0.302649, 1.299481, -1.108949, -0.903879, 1.565521, -0.358079, -1.058819, -1.517569, -1.295389, 0.281951, -0.509059, 0.697771, -0.734609, 0.040191, 0.022531,
        -1.040849, -0.725799, -0.167349, -1.314209, -0.880349, -0.895789, -0.784399, 0.128421, -0.491839, -0.181679, -0.886529, -0.921859, -0.706469, -0.383429, -0.365549, -0.290809, -0.792349, 1.743091, -0.862909, -1.012339,
        -1.146209, -0.740619, 0.903841, -0.909969, -1.155219, -0.045769, 0.613981, -0.599989, -1.641279, -0.952429, -1.279909, -1.555439, 0.004481, -1.182289, -1.555709, -0.942449, -1.412829, -1.641509, 1.885531, -1.188139,
        -0.957339, -0.992399, -1.061769, -0.738929, -0.901229, 0.151281, 0.306761, -0.133419, -1.063869, 1.516311, -inf, 0.553745, -0.395165, -0.556055, -0.348525, -0.361295, -0.532395, -0.715345, -0.349015, -0.753825,
        -0.699475, -0.481695, -1.108375, 0.919365, -0.858755, -0.250405, -0.502055, -0.639535, 0.621335, -1.344655, -0.209265, 0.696285, 0.696975, -0.743065, -0.389395, -0.917985, -0.503065, -0.314025, 0.740925, -1.050755,
        -0.416935, -0.348875, 0.327105, 0.647255, 0.051985, -0.053845, 0.166465, 0.695445, -0.486845, -0.302565, 1.134615, 0.158805, -0.444175, -0.946045, -0.695475, 0.337165, 0.006665, 0.634705, -0.157465, 0.430525,
        0.462265, -0.412915, -0.172725, 0.416365, -0.726855, -0.369645, -0.081325, -0.279615, -0.018385, 0.082385, 0.495855, -0.311845, -0.251365, -0.161515, -0.162065, -0.137605, -0.066405, -0.208835, 0.521795, -0.253965,
        -0.538415, -0.614275, -0.132675, 0.741435, -0.234475, -0.658695, 0.099875, 0.688375, -0.035285, -1.192325, -0.330455, -1.184745, -1.164625, 0.097845, -0.820715, -1.056165, -0.844595, -0.860675, -1.107545, 0.429095,
        -0.640345, -0.392375, -0.387265, -0.499235, -0.163715, -0.301475, 0.444505, 0.653235, 0.316175, -0.560995, 0.682245, -inf, -0.608328, 0.128082, -0.059158, 0.661962, 0.182762, 0.351862, -0.134828, 0.306962,
        0.031972, 0.666612, -0.190508, 0.331432, -1.245948, -0.430348, 0.241792, -0.295318, -0.456378, -0.863278, -0.734158, -0.104318, -1.323368, -1.475968, -0.149638, 0.411812, -0.319428, 0.327172, 0.406582, 0.110332,
        0.385352, -0.308718, 0.328152, 0.004392, -1.333528, 0.150832, 0.248352, -0.042578, -1.526198, 0.561882, 0.326202, -1.505898, -0.016668, 0.620432, 0.470852, 0.745102, -1.363488, 0.123752, -0.708918, 0.231592,
        -0.128208, -0.217898, 0.506612, 0.389802, -0.033508, 0.385232, 0.961652, 0.309462, 0.149522, -0.870838, 0.141482, -0.129728, 0.704342, 0.508682, 0.323512, 0.502572, 0.226122, 0.171212, 0.282662, -1.348418,
        0.731472, -0.328868, -0.088268, 0.297022, -0.804908, 0.317522, -0.129458, 0.160732, -0.660598, 0.157822, 0.156712, 0.714002, -1.607598, -0.748698, -1.582408, -0.675518, -0.511478, -1.172918, -0.383738, -0.490038,
        -1.667948, 0.018872, 0.312502, 0.260642, -0.191738, 0.130682, 0.233092, -0.277098, -0.410858, 0.263212, 0.275242, -1.281048, -inf, -1.123408, -0.793108, -0.933168, -0.825478, -0.837708, -0.798408, -1.096718,
        -0.481828, -0.800158, -0.634688, -0.762128, -0.924348, -1.600908, -1.239158, -0.861988, -0.729668, -0.894928, -0.664438, -1.134828, -1.045808, -1.706228, -1.794078, -1.091578, -0.821888, -0.869308, -0.815808, 0.686272,
        -0.832518, -0.907028, -0.499988, -0.810328, -0.872268, -1.714958, -0.828998, -0.781338, -0.881278, -1.830788, -0.797718, -0.788548, -1.795678, -0.862388, -0.785808, -0.860998, -0.580718, -1.754248, -0.834758, -1.148898,
        -0.786458, -0.658468, 0.299642, -0.641098, -0.365208, -0.055078, -0.370258, -0.632878, -0.695338, 0.895432, -1.510788, -0.726568, 0.080612, 0.033602, -0.611548, -0.293088, -0.270858, -0.555668, -0.107478, -0.523338,
        -1.693268, -0.776838, -0.792328, -1.081288, -0.734278, -0.967198, -0.384578, -1.101118, -0.334168, 0.418272, -0.818488, -1.103498, -0.456308, -1.854098, -1.460988, -1.923768, -1.125288, -1.012998, -1.852648, -0.891178,
        -0.984078, -1.919718, -1.020348, -0.952988, -0.872368, -0.883168, -0.859678, -0.842278, -0.075078, -0.735558, -0.928418, -0.862468, -1.655418, -inf, -0.459402, 0.819658, 0.164998, 0.397648, 0.434768, 0.451938,
        0.252128, 0.772398, -0.174282, 0.376528, -0.289182, 0.055478, -0.994442, -0.360552, 1.262668, -0.451752, -0.675312, -0.642582, -0.979602, 0.220048, -1.053712, -1.190012, 2.399828, 0.684818, -0.580522, 0.967398,
        0.498478, 0.783118, 0.131328, -0.452902, 0.451208, 0.330208, -1.073742, 0.512398, 0.727288, 0.161768, -1.229382, 0.879548, 0.563388, -1.127802, 0.210188, 0.432948, 0.272368, 0.360488, -1.238712, 0.440368,
        -0.561012, 0.947378, 0.067848, -0.034752, 1.023048, 0.592118, 0.246778, 0.315358, 0.553748, 1.046138, 0.637798, -0.738332, 1.299528, 0.055808, 0.908448, 0.741758, 0.621598, 0.471818, 0.445268, 0.607958,
        0.573548, -1.119532, 0.595308, -0.534462, 0.516718, 0.719408, -0.630832, 1.138468, 0.291638, 0.548438, -0.478042, 0.800318, -0.088992, 0.459588, -1.590422, -0.276422, -1.536572, -0.851372, -0.763732, -1.215792,
        -0.600572, -0.810332, -1.404422, 2.107578, 1.811498, 0.463588, -0.219262, 0.528678, 0.374078, -0.109632, -0.239822, 1.611638, 0.374508, -1.009582, -inf, -0.726558, 1.196402, 1.438832, 1.147422, 0.811402,
        0.081892, 0.589482, 0.099862, -0.652668, -0.232108, -0.587508, -0.735008, -1.302228, -0.384748, 0.487222, -0.746758, -0.896448, -0.884678, -1.188218, 1.253502, -1.278158, -1.495978, -0.036838, 0.539502, -0.874548,
        -0.008338, 0.051532, -0.051458, -0.651638, -0.720968, -0.126928, -0.174758, -1.268888, 0.847502, 0.134082, 0.208262, -1.510558, 0.087812, 0.072532, -1.483458, -0.325058, -0.062798, -0.470518, -0.251718, -1.469718,
        0.009312, -0.801758, 0.306552, -0.128878, -0.382208, 0.019072, 0.082552, -0.214218, -0.322288, 0.036902, 0.091722, 0.730922, -0.847468, 0.023602, -0.305658, 0.052492, 0.103922, 0.031932, 0.072412, 0.032632,
        0.107772, 0.106032, -1.258648, 0.113332, -0.788378, 1.733842, 0.400362, -0.890968, 0.277202, 1.960982, 0.368252, -0.779378, 0.329682, -0.846918, 0.042412, -1.578568, 2.469332, -1.741308, -1.005928, -1.013638,
        -1.420388, -0.881948, -1.119688, -1.556528, 0.361682, 0.304672, 0.739162, -0.577478, -0.066888, 0.109872, -0.306168, -0.550878, -0.054218, -0.137478, -1.293278, -inf, -0.208327, 0.014653, -0.175877, 0.224763,
        0.329303, 0.140533, -0.487207, 0.144563, 0.153343, 0.064133, 0.944203, -0.133407, -1.035617, -0.714277, -0.068817, 1.710113, -0.012877, -0.738227, -0.522827, -0.317117, -1.160837, -1.298197, -0.477207, -0.010777,
        -0.010097, -0.030177, -0.003237, 0.014643, -0.250197, 0.138003, 0.131293, -0.067087, -1.231997, -0.010097, 0.268383, 0.196013, -1.358917, 0.006063, 0.169153, -1.345877, 0.394743, 0.122663, -0.121737, 0.031473,
        -1.165757, 0.029963, -0.447557, 0.264253, 0.021833, -0.158887, 0.153293, 0.145333, -0.023337, -0.037637, 0.125043, 0.165393, -0.007287, -0.928317, -0.002007, 0.247783, 0.106503, 0.218033, 0.134513, 0.330473,
        0.118903, 0.103823, 0.211433, -1.264247, 0.603343, 0.245303, -0.448067, 0.147913, -0.557837, 0.098933, -0.480747, 0.302343, -0.530497, 0.055483, -0.476857, 0.278463, -1.754267, -1.024257, -1.328897, -0.278727,
        2.130533, -1.397327, 1.967163, -0.077807, -1.565647, 0.005733, -0.170517, 0.366133, 0.351343, 0.836113, 0.829113, -0.252847, -0.294887, -0.118337, 0.403663, -1.079127, -inf, 0.054210, 0.014130, -0.181460,
        0.264700, -0.028730, -0.064370, -0.455170, 0.232910, -0.045230, -0.129360, 1.626920, -0.453890, -0.450580, -0.725040, -0.027590, 0.168000, -0.074750, -0.348190, -0.609510, -0.217620, -0.667740, -0.627860, -0.503210,
        -0.013970, -0.113750, -0.078150, -0.004640, 0.086650, -0.251950, 0.094260, -0.025660, 0.187740, -0.778100, 0.102960, 0.552700, 0.078460, -0.675100, -0.038110, 0.116400, -0.725900, 0.142990, 0.056600, -0.327240,
        -0.173750, -0.336550, 0.107950, 0.009220, 0.344810, 0.734170, 0.064260, 0.034040, 0.497540, 0.512290, -0.144670, 0.134490, 0.378110, -0.022930, -0.728820, 0.197910, 0.104400, 0.134060, 0.106290, 0.082680,
        0.199450, 0.459430, 0.244930, 0.132710, -0.848060, 0.093330, 0.089260, -0.379210, 0.597390, -0.097840, 0.427080, -0.431270, 0.231720, -0.119720, 0.126020, -0.647030, 0.069480, -1.650880, -0.960620, -0.637400,
        2.349760, -0.225050, -1.378710, 0.331150, -0.192930, -1.100080, -0.289550, -0.147010, -0.048360, -0.013790, 0.046720, 0.320020, 0.110490, 0.146350, -0.029980, -0.109350, -0.542570, -inf, 1.038586, -0.768344,
        -0.791044, -0.648184, -0.686474, -0.880534, -1.034584, -0.572164, -0.755524, -1.026794, -0.449064, -1.223804, 1.093096, -1.103614, -0.577534, -0.456904, -0.410554, 0.394886, -1.261554, -0.513304, 0.550426, 0.869446,
        -0.996114, -0.758494, -0.790394, -0.643904, -0.622784, -0.213444, -1.199974, -0.098854, -0.542044, -0.027904, 0.205026, -0.089134, -0.510354, -0.050804, 0.643186, -0.842404, -0.586034, 0.267056, 0.113556, -0.767124,
        -1.235334, -1.013954, 1.108556, 0.157446, 0.715816, -0.399654, 0.165286, 0.516866, -0.659214, -0.489904, 0.055306, -1.018624, -0.639224, -0.515914, -0.584844, -0.366734, -0.211794, 0.492696, -0.531684, -0.713424,
        -0.183334, -0.381154, -0.439714, -0.344954, -0.425094, 0.097516, -0.580014, -0.264334, -1.042754, -0.616994, 0.470096, -0.635624, -1.006864, -0.307184, 0.792196, -0.232734, -1.362264, -0.624644, -1.521534, -1.476124,
        2.002356, -0.615884, -0.929544, -1.154344, -0.745154, -0.937364, -0.100884, -0.947564, -0.718664, -0.725384, -0.209514, -0.256394, -0.604744, 0.533476, 0.578676, -0.331854, -0.397984, 0.339946, -inf, -0.255552,
        -0.974792, -1.060782, -0.985722, -0.987162, -1.171322, -1.085762, -1.014462, -1.357152, -1.330702, -1.164092, -1.549092, -0.613652, -0.489552, -0.880932, -1.182612, -1.261532, -0.619932, -1.355412, -0.286892, -0.768502,
        -0.855462, -1.053632, -1.013572, -1.376682, -1.144932, -0.979572, -0.673062, -1.590742, -1.166142, -1.025722, -0.006902, -0.669422, -0.762052, -0.885042, -0.558402, -0.854622, -1.128972, -0.979802, -0.665512, -0.577902,
        -1.105482, -1.542162, -1.326862, -1.087442, -0.704232, 1.378908, -0.849462, 0.347518, -0.472262, -1.075522, -0.915862, -0.504592, -1.358332, -1.038052, -0.954002, -0.922692, 3.438988, -0.720352, -0.455422, -0.986442,
        -1.026472, -0.681392, -0.855812, -0.834622, -0.770042, -0.951412, -0.762502, -0.936262, -1.247142, -1.049302, -0.952232, 0.529668, -0.979632, -1.070662, -0.757502, -0.489072, -0.743772, -1.487902, -1.009152, 4.168118,
        -1.293432, -1.349232, -1.400222, -1.464782, 0.340618, -1.491822, -1.630472, -0.942532, -1.167472, -0.914692, -1.036712, -1.181812, -0.860522, -0.985372, -0.325622, -0.333952, -0.753332, -1.217132, -0.164952, -inf,
        0.265145, -0.681195, -0.821295, -0.655215, -0.656505, -0.742425, -0.888085, -0.611195, -1.035945, -0.855965, -0.915855, -1.049525, -0.463645, 0.044685, -0.558495, -0.969245, -1.074375, -0.422625, -1.431325, 1.460075,
        -0.573805, -0.744305, -0.748995, -0.655055, -1.204185, -0.730625, -0.613305, -0.364945, -1.077175, -0.936055, -0.644955, -0.044905, -0.390755, -0.441645, -0.518015, -0.291325, -0.717855, -0.712285, -0.595625, -0.536585,
        -0.325705, -0.690205, -1.016035, -0.856365, -0.872805, -0.392215, -0.012555, -0.407215, -0.151705, -0.224975, -0.657775, -0.535845, -0.234215, -0.889555, -0.627855, -0.563495, -0.601825, 0.961575, -0.401205, -0.201175,
        -0.591715, -0.616695, -0.494255, -0.494095, -0.478945, -0.439025, -0.563485, -0.526825, -0.555585, -1.006805, -0.874285, -0.566355, -0.000795, -0.593895, -0.878745, -0.429115, -0.288445, 0.070595, -0.456925, 0.326315,
        0.196165, -1.222605, -1.085675, -1.215955, -1.204285, 3.054375, -1.249345, -1.467725, -0.747435, -0.844395, 0.415035, -0.673735, -0.908895, -0.517875, -0.620725, -0.100855, 0.105295, -0.418295, -0.785835, -0.517265,
    };

    namespace sycl = cl::sycl;
    using target = sycl::access::target;
    using mode = sycl::access::mode;
    {
        auto exception_handler = [](const sycl::exception_list& exceptions) {
          for (const std::exception_ptr& e : exceptions) {
              try {
                  std::rethrow_exception(e);
              } catch (const sycl::exception& e) {
                  std::cout << "Caught asynchronous SYCL exception:" << '\n' << e.what() << '\n';
              }
          }
        };

        auto queue = sycl::queue(sycl::default_selector(), exception_handler);
        auto dp = sycl::buffer<float, 2>(sycl::range<2>(2, 106));
        auto emissions_buf =
            sycl::buffer<float, 1>(emission_scores.data(), sycl::range<1>(2020),
                                   sycl::property::buffer::use_host_ptr());

        auto max_M_buf = sycl::buffer<float, 1>(sycl::range<1>(100));

        try {
            // dp initialization
            queue.submit([&](sycl::handler& cgh) {
              auto dpA = dp.get_access<mode::discard_write, target::global_buffer>(cgh);
              cgh.parallel_for<init_dp_spec_100>(sycl::range<1>(106), [=](sycl::item<1> col_work_item) {
                dpA[0][col_work_item.get_linear_id()] = -inf;
              });
            });

            queue.submit([&](sycl::handler& cgh) {
              auto dpA = dp.get_access<mode::write, target::global_buffer>(cgh);
              cgh.single_task<init_N_B_spec_100>([=]() {
                dpA[1][0] = -inf;
                dpA[0][104] = 0.0;
                dpA[0][105] = move_score; // tr_N_B
              });
            });

            size_t cur_row = 1;
            size_t prev_row = 0;

            for (size_t i = 1; i < seq.size(); ++i) {
                const auto stride = amino_acid_num.at(seq[i]) * 20;

                // Calculate M states
                queue.submit([&](sycl::handler& cgh) {
                  auto dpA = dp.get_access<mode::write, target::global_buffer>(cgh);
                  auto emissions_bufA = emissions_buf.get_access<mode::read, target::constant_buffer>(cgh);

                  cgh.parallel_for<M_states_handler_spec_100>(
                      sycl::range<1>(100), [=](sycl::item<1> col_work_item) {
                        auto cur_col = col_work_item.get_linear_id() + 1;
                        dpA[cur_row][cur_col] =
                            emissions_bufA[stride + cur_col] +
                            sycl::fmax(dpA[prev_row][cur_col - 1], dpA[prev_row][105] + static_cast<float>(-8.546946149565585));
                      });
                });

                // Data preparation to get max from M states
                queue.submit([&](sycl::handler& cgh) {
                  auto dpA = dp.get_access<mode::read, target::global_buffer>(cgh);
                  auto bufA = max_M_buf.get_access<mode::discard_write, target::global_buffer>(cgh);

                  cgh.parallel_for<copy_M_spec_100>(sycl::range<1>(100), [=](sycl::item<1> col_work_item) {
                    auto id = col_work_item.get_linear_id();
                    bufA[id] = dpA[cur_row][id + 1];
                  });
                });

                auto left_half_size = 50;

                while (left_half_size > 1) {
                    queue.submit([&](sycl::handler& cgh) {
                      auto bufA = max_M_buf.get_access<mode::read_write, target::global_buffer>(cgh);

                      cgh.parallel_for<reduction_step_spec_100>(sycl::range<1>(left_half_size), [=](sycl::item<1> item) {
                        auto id = item.get_linear_id();
                        auto offset = left_half_size + id;
                        bufA[id] = sycl::fmax(bufA[id], bufA[offset]);
                      });
                    });

                    left_half_size += (left_half_size % 2);
                    left_half_size /= 2;
                }

                queue.submit([&](sycl::handler& cgh) {
                  auto dpA = dp.get_access<mode::read_write, target::global_buffer>(cgh);
                  auto bufA = max_M_buf.get_access<mode::read, target::global_buffer>(cgh);
                  cgh.single_task<E_J_C_N_B_states_handler_spec_100>([=]() {
                    dpA[cur_row][101] = sycl::fmax(bufA[0], bufA[1]);
                    dpA[cur_row][102] = sycl::fmax(dpA[prev_row][102] + loop_score, dpA[cur_row][101] + static_cast<float>(-0.6931471805599453));
                    dpA[cur_row][103] = sycl::fmax(dpA[prev_row][103] + loop_score, dpA[cur_row][101] + static_cast<float>(-0.6931471805599453));
                    dpA[cur_row][104] = dpA[prev_row][104] + loop_score;
                    dpA[cur_row][105] = sycl::fmax(dpA[cur_row][104] + move_score, dpA[cur_row][102] + move_score);
                  });
                });

                prev_row = cur_row;
                cur_row = 1 - cur_row;
            }

            auto dpA_host = dp.get_access<mode::read>();
            return dpA_host[prev_row][103] + move_score;
        } catch (sycl::exception& e) {
            std::cout << e.what() << '\n';
        }
    }
    return 0;
}


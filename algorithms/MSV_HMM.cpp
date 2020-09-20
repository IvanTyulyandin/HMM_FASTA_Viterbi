#include "MSV_HMM.hpp"

#include <CL/cl2.hpp>
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

std::string_view get_error_string(cl_int error) {
    switch(error) {
        // run-time and JIT compiler errors
        case 0: return "CL_SUCCESS";
        case -1: return "CL_DEVICE_NOT_FOUND";
        case -2: return "CL_DEVICE_NOT_AVAILABLE";
        case -3: return "CL_COMPILER_NOT_AVAILABLE";
        case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case -5: return "CL_OUT_OF_RESOURCES";
        case -6: return "CL_OUT_OF_HOST_MEMORY";
        case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case -8: return "CL_MEM_COPY_OVERLAP";
        case -9: return "CL_IMAGE_FORMAT_MISMATCH";
        case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case -11: return "CL_BUILD_PROGRAM_FAILURE";
        case -12: return "CL_MAP_FAILURE";
        case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
        case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case -15: return "CL_COMPILE_PROGRAM_FAILURE";
        case -16: return "CL_LINKER_NOT_AVAILABLE";
        case -17: return "CL_LINK_PROGRAM_FAILURE";
        case -18: return "CL_DEVICE_PARTITION_FAILED";
        case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

        // compile-time errors
        case -30: return "CL_INVALID_VALUE";
        case -31: return "CL_INVALID_DEVICE_TYPE";
        case -32: return "CL_INVALID_PLATFORM";
        case -33: return "CL_INVALID_DEVICE";
        case -34: return "CL_INVALID_CONTEXT";
        case -35: return "CL_INVALID_QUEUE_PROPERTIES";
        case -36: return "CL_INVALID_COMMAND_QUEUE";
        case -37: return "CL_INVALID_HOST_PTR";
        case -38: return "CL_INVALID_MEM_OBJECT";
        case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case -40: return "CL_INVALID_IMAGE_SIZE";
        case -41: return "CL_INVALID_SAMPLER";
        case -42: return "CL_INVALID_BINARY";
        case -43: return "CL_INVALID_BUILD_OPTIONS";
        case -44: return "CL_INVALID_PROGRAM";
        case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
        case -46: return "CL_INVALID_KERNEL_NAME";
        case -47: return "CL_INVALID_KERNEL_DEFINITION";
        case -48: return "CL_INVALID_KERNEL";
        case -49: return "CL_INVALID_ARG_INDEX";
        case -50: return "CL_INVALID_ARG_VALUE";
        case -51: return "CL_INVALID_ARG_SIZE";
        case -52: return "CL_INVALID_KERNEL_ARGS";
        case -53: return "CL_INVALID_WORK_DIMENSION";
        case -54: return "CL_INVALID_WORK_GROUP_SIZE";
        case -55: return "CL_INVALID_WORK_ITEM_SIZE";
        case -56: return "CL_INVALID_GLOBAL_OFFSET";
        case -57: return "CL_INVALID_EVENT_WAIT_LIST";
        case -58: return "CL_INVALID_EVENT";
        case -59: return "CL_INVALID_OPERATION";
        case -60: return "CL_INVALID_GL_OBJECT";
        case -61: return "CL_INVALID_BUFFER_SIZE";
        case -62: return "CL_INVALID_MIP_LEVEL";
        case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
        case -64: return "CL_INVALID_PROPERTY";
        case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
        case -66: return "CL_INVALID_COMPILER_OPTIONS";
        case -67: return "CL_INVALID_LINKER_OPTIONS";
        case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

        // extension errors
        case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
        case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
        case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
        case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
        case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
        case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
        default: return "Unknown OpenCL error";
    }
}

void check_errors(cl_int res, std::string_view where) {
    std::cout << where << "... ";
    if (res != CL_SUCCESS) {
        std::cout << get_error_string(res) << '\n';
    } else {
        std::cout << "OK\n";
    }
}

} // namespace

MSV_HMM::MSV_HMM(const Profile_HMM& base_hmm) : model_length(base_hmm.model_length) {
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

Log_score MSV_HMM::parallel_run_on_sequence(const Protein_sequence& seq) {
    auto err = cl_int(CL_SUCCESS);

    auto platforms = std::vector<cl::Platform>();
    err = cl::Platform::get(&platforms);
    check_errors(err, "platform detection");

    auto platform = platforms[0];

    std::cout << "OpenCL version for platform \"" << platform.getInfo<CL_PLATFORM_NAME>()
           << "\" is \"" << platform.getInfo<CL_PLATFORM_VERSION>() << "\"\n";

    auto devices = std::vector<cl::Device>();
    err = platform.getDevices(CL_DEVICE_TYPE_DEFAULT, &devices);
    check_errors(err, "get devices");
    for (const auto& device: devices) {
        std::cout << device.getInfo<CL_DEVICE_NAME>() << '\n';
    }

    auto ctx = cl::Context(devices, NULL, NULL, NULL, &err);
    check_errors(err, "context creation");

    return seq.size();
}

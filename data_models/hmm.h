#pragma once

#include <array>
#include <string>
#include <vector>

// info about format was taken from page 210 of
// http://eddylab.org/software/hmmer/Userguide.pdf

constexpr int NUM_OF_AMINO_ACIDS = 20;
constexpr int NUM_OF_TRANSITIONS = 7;

template <int N>
using Probabilities_array = std::array<float, N>;

template <int N>
using Probabilities_arrays_vector = std::vector<Probabilities_array<N>>;

class Hmm {
public:

    explicit Hmm(const std::string& file_path);

    Probabilities_arrays_vector<NUM_OF_AMINO_ACIDS> match_emissions;
    Probabilities_arrays_vector<NUM_OF_AMINO_ACIDS> insert_emissions;
    Probabilities_arrays_vector<NUM_OF_TRANSITIONS> transitions;
    size_t model_length;

    // mu and lambda for Gumbel distributions
    // for MSV and Viterbi scores
    float stats_local_msv_mu;
    float stats_local_msv_lambda;
    float stats_local_viterbi_mu;
    float stats_local_viterbi_lambda;

    // theta and lambda
    // for exponential tails for Forward scores
    float stats_local_forward_theta;
    float stats_local_forward_lambda;

private:
    void extract_length(std::ifstream& file);
    void extract_stats_local(std::ifstream& file);
    void extract_probabilities(std::ifstream& file);
};

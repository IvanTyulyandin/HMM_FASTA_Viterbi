#pragma once

#include <array>
#include <string>
#include <vector>

// info about format was taken from page 210 of
// http://eddylab.org/software/hmmer/Userguide.pdf

constexpr int NUM_OF_AMINO_ACIDS = 20;

using Neg_ln_probabilities = std::vector<std::array<float, NUM_OF_AMINO_ACIDS>>;

class Hmm {
public:

    explicit Hmm(const std::string& file_path);

    Neg_ln_probabilities transition;
    Neg_ln_probabilities emission;
    size_t length;

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
};

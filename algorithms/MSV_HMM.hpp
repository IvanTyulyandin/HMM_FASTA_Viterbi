#pragma once

#include "FASTA_protein_sequences.hpp"
#include "Profile_HMM.hpp"

// Sequential MSV implementation
// https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3197634/

using Log_score = float;

template <int N> using Log_scores_array = std::array<Log_score, N>;

template <int N> using Log_scores_arrays_vector = std::vector<Log_scores_array<N>>;

class MSV_HMM {
  public:
    explicit MSV_HMM(const Profile_HMM& base_hmm);

    Log_score run_on_sequence(const Protein_sequence& seq);

    Log_score parallel_run_on_sequence(const Protein_sequence& seq);

  private:
    Profile_name name;
    size_t model_length;
    // 2D matrix emulation [NUM_OF_AMINO_ACIDS][hmm model length]
    std::vector<Log_score> emission_scores;

    // tr_X_Y == transition scores from HMM state X to state Y
    // All of these scores should be considered as weights of HMM
    // rather than probabilities.
    Log_score tr_loop; // tr_N_N == tr_C_C == tr_J_J
    Log_score tr_move; // tr_N_B == tr_C_T == tr_J_B
    Log_score tr_B_Mk;
    Log_score tr_E_C;
    Log_score tr_E_J;

    void init_transitions_depend_on_seq(const Protein_sequence& seq);
};

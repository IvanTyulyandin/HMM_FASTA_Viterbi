#pragma once

#include "FASTA_protein_sequences.hpp"

using Log_score = float;

class MSV_HMM_spec {
  public:
    static Log_score parallel_run_on_sequence_100(const Protein_sequence& seq);
};

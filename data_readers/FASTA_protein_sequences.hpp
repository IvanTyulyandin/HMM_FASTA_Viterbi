#pragma once

#include <string>
#include <vector>

using Protein_sequence = std::string;
using Protein_sequences = std::vector<Protein_sequence>;

class FASTA_protein_sequences {
  public:
    explicit FASTA_protein_sequences(const std::string& file_path);

    Protein_sequences sequences;
};

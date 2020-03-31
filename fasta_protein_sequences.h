#pragma once

#include <string>
#include <vector>

class Fasta_protein_sequences {
public:
    explicit Fasta_protein_sequences(const std::string& file_path);

    using Protein_sequences = std::vector<std::string>;
    Protein_sequences sequences;
};

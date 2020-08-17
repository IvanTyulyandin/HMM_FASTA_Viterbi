#include "FASTA_protein_sequences.hpp"

#include <fstream>
#include <iostream>
#include <unordered_set>
#include <algorithm>


FASTA_protein_sequences::FASTA_protein_sequences(const std::string& file_path) {
    auto file = std::ifstream(file_path);
    if (file.fail()) {
        std::cout << "Failed to open " << file_path << '\n';
        return;
    }

    auto line = std::string();

    while (std::getline(file, line)) {
        if (line[0] == '>') {
            sequences.push_back("#");
        } else {
            sequences.back() += line;
        }
    }

    auto allowed_symbols = std::unordered_set<char>(
        {'#', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'});

    sequences.erase(
        std::remove_if(sequences.begin(), sequences.end(),
            [&](Protein_sequence& protein) {
                for (auto amino_acid : protein) {
                    if (allowed_symbols.find(amino_acid) == allowed_symbols.end()) {
                        std::cout << "Warning: sequence " << protein << " was rejected.\nReason: prohibited symbol "
                            << amino_acid << " in " << file_path << " FASTA file\n\n";
                        return true;
                    }
                }
                return false;
            }),
        sequences.end());

    file.close();
}

#include "fasta_protein_sequences.h"

#include <fstream>
#include <iostream>

Fasta_protein_sequences::Fasta_protein_sequences(const std::string& file_path) {
    auto file = std::ifstream(file_path);
    if (file.fail()) {
        std::cout << "Failed to open " << file_path << '\n';
        return;
    }

    auto line = std::string();
    auto current_sequence = std::string();

    while (std::getline(file, line)) {
        if (line[0] == '>') {
            sequences.push_back("");
        } else {
            sequences.back() += line;
        }
    }

    file.close();
}

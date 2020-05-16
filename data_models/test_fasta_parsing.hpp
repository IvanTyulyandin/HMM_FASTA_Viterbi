#pragma once

#include "FASTA_protein_sequences.hpp"

#include <cassert>

void test_fasta_parsing() {
    auto fasta_seq = FASTA_protein_sequences("../fasta_like_example.fsa");
    assert((fasta_seq.sequences == Protein_sequences {
            {"ACDEFGHIKLMNPQTVWY"},
            {"ACDKLMNPQTVWYEFGHI"},
            {"EFMNRGHIKLMNPQT"},
            {"MKMRFFSSPCGKAAVDPADRCKEVQQIRDQHPSKIPVIIERYKGEKQLPVLDKTKFLVPDHVNMSELVKI"
             "IRRRLQLNPTQAFFLLVNQHSMVSVSTPIADIYEQEKDEDGFLYMVYASQETFGFIRENE"}
    }));
}

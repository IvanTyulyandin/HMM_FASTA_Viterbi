#pragma once

#include "fasta_protein_sequences.h"

#include <cassert>

void test_fasta_parsing() {
    auto fasta_seq = Fasta_protein_sequences("../fasta_like_example.fsa");
    assert((fasta_seq.sequences == Fasta_protein_sequences::Protein_sequences{
            {"ACDEFGHIKLMNPQTVWY"},
            {"ACDKLMNPQTVWYEFGHI"},
            {"EFMNRGHIKLMNPQT"},
            {"MKMRFFSSPCGKAAVDPADRCKEVQQIRDQHPSKIPVIIERYKGEKQLPVLDKTKFLVPDHVNMSELVKI"
             "IRRRLQLNPTQAFFLLVNQHSMVSVSTPIADIYEQEKDEDGFLYMVYASQETFGFIRENE"}
    }));
}

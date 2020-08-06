#include "FASTA_protein_sequences.hpp"

#include <cassert>

int main() {
    auto fasta_seq = FASTA_protein_sequences("../fasta_like_example.fsa");
    assert(
        (fasta_seq.sequences == Protein_sequences{{"#ACDEFGHIKLMNPQTVWY"},
                                                  {"#ACDKLMNPQTVWYEFGHI"},
                                                  {"#EFMNRGHIKLMNPQT"},
                                                  {"#MKMRFFSSPCGKAAVDPADRCKEVQQIRDQHPSKIPVIIERYKGEKQLPVLDKTKFLVPDHVNMS"
                                                   "E"
                                                   "LVKI"
                                                   "IRRRLQLNPTQAFFLLVNQHSMVSVSTPIADIYEQEKDEDGFLYMVYASQETFGFIRENE"}}));
}

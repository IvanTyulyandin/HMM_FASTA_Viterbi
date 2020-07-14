#include "FASTA_protein_sequences.hpp"
#include "MSV_HMM.hpp"

#include <cassert>

int main() {
    // Test for out-of-bound access.
    // If didn't fail with an exception or segfault, the test is passed
    auto msv = MSV_HMM(Profile_HMM("../100.hmm"));
    auto fasta = FASTA_protein_sequences("../fasta_like_example.fsa");
    msv.run_on_sequence(fasta.sequences.front());
    msv.parallel_run_on_sequence(fasta.sequences.front());
}
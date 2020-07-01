#include "test_MSV.hpp"
#include "test_fasta_parsing.hpp"
#include "test_hmm_parsing.hpp"

int main() {
#ifndef NDEBUG
    test_hmm_parsing();
    test_fasta_parsing();
    test_MSV();
#endif
}

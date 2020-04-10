#include "test_hmm_parsing.h"
#include "test_fasta_parsing.h"

int main() {
#ifndef NDEBUG
    test_hmm_parsing();
    test_fasta_parsing();
#endif
}

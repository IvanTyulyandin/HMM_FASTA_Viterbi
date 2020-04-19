#include "test_hmm_parsing.h"
#include "test_fasta_parsing.h"
#include "test_MSV.h"

int main() {
#ifndef NDEBUG
    test_hmm_parsing();
    test_fasta_parsing();
    test_MSV();
#endif
}

#include <iostream>
#include "hmm.h"

int main() {
    auto hmm = Hmm("100.hmm");
    std::cout << hmm.length << ' ' << hmm.stats_local_forward_theta << ' ' << hmm.stats_local_forward_lambda << '\n';
}

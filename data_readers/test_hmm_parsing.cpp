#include "Profile_HMM.hpp"

#include <cassert>
#include <cmath>

// code below was taken from
// https://en.cppreference.com/w/cpp/types/numeric_limits/epsilon
template <class T>
typename std::enable_if<!std::numeric_limits<T>::is_integer, bool>::type almost_equal(T x, T y, int ulp = 5) {
    // the machine epsilon has to be scaled to the magnitude of the values used
    // and multiplied by the desired precision in ULPs (units in the last place)
    return std::fabs(x - y) <= std::numeric_limits<T>::epsilon() * std::fabs(x + y) * ulp
           // unless the result is subnormal
           || std::fabs(x - y) < std::numeric_limits<T>::min();
}

float neg_ln_to_prob(double data) { return std::exp(-1 * static_cast<float>(data)); }

int main() {
    auto hmm = Profile_HMM("../profile_HMMs/100.hmm");

    // check header parsing
    assert(hmm.model_length == 101);
    assert(hmm.name == "Pfam-B_229");
    assert(almost_equal(hmm.stats_local_msv_mu, static_cast<float>(-9.5678)));
    assert(almost_equal(hmm.stats_local_forward_lambda, static_cast<float>(0.71755)));

    assert(almost_equal(hmm.insert_emissions[0][0], neg_ln_to_prob(2.68618)));
    assert(almost_equal(hmm.transitions[0][6], neg_ln_to_prob(0.0)));
    assert(almost_equal(hmm.match_emissions[1][0], neg_ln_to_prob(2.66211)));
    assert(almost_equal(hmm.match_emissions[100][19], neg_ln_to_prob(4.01014)));
    assert(almost_equal(hmm.insert_emissions[1][19], neg_ln_to_prob(3.61503)));
    assert(almost_equal(hmm.transitions[1][1], neg_ln_to_prob(4.09464)));
    assert(almost_equal(hmm.insert_emissions[100][19], neg_ln_to_prob(3.61503)));
    assert(almost_equal(hmm.transitions[100][5], neg_ln_to_prob(0.0)));
    assert(almost_equal(hmm.transitions[100][6], neg_ln_to_prob(0.0)));
}

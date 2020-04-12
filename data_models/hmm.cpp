#include "hmm.h"

#include <cmath>
#include <fstream>
#include <iostream>

namespace {
    void skip_next_words(std::string_view& data_view, int times) {
        for (int i = 0; i < times; ++i) {
            data_view.remove_prefix(data_view.find_first_of(' '));
            data_view.remove_prefix(data_view.find_first_not_of(' '));
        }
    }

    std::string read_value_after_tag(std::istream& file, std::string_view tag) {
        auto line = std::string();
        while (std::getline(file, line)) {
            auto line_view = std::string_view {line};
            line_view.remove_prefix(line_view.find_first_not_of(' '));
            if (line_view.substr(0, tag.size()) == tag) {
                skip_next_words(line_view, 1);
                return std::string {line_view};
            }
        }
        return {};
    }


    void parse_two_floats_after_name(std::string_view data, float& fst, float& snd) {
        skip_next_words(data, 1);
        char* rest_of_line = nullptr;
        fst = std::strtof(data.data(), &rest_of_line);
        snd = std::strtof(rest_of_line, nullptr);
    }


    template <int N>
    Probabilities_array<N> parse_probabilities(std::string_view floats_view) {
        auto res = Probabilities_array<N>();
        floats_view.remove_prefix(floats_view.find_first_not_of(' '));

        for (auto i = 0; i < N; ++i) {
            res[i] = std::exp(-1 * std::strtof(floats_view.data(), nullptr));
            skip_next_words(floats_view, 1);
        }

        return res;
    }
}


Hmm::Hmm(const std::string& file_path) {
    auto file = std::ifstream(file_path);
    if (file.fail()) {
        std::cout << "Failed to open " << file_path << '\n';
        return;
    }
    extract_length(file);
    extract_stats_local(file);
    extract_probabilities(file);

    file.close();
}


void Hmm::extract_length(std::ifstream& file) {
    // expecting LENG x
    model_length = std::stoi(read_value_after_tag(file, "LENG"));
}


void Hmm::extract_stats_local(std::ifstream& file) {
    // expecting 3 lines STATS LOCAL x float float
    // where x can be MSV, VITERBI and FORWARD
    for (int i = 0; i < 3; ++i) {
        auto data = read_value_after_tag(file, "STATS");
        auto data_view = std::string_view {data};
        // skip LOCAL
        skip_next_words(data_view, 1);

        switch (data_view[0]) {
            case 'M': // MSV
                parse_two_floats_after_name(data_view, stats_local_msv_mu, stats_local_msv_lambda);
                break;
            case 'V': // Viterbi
                parse_two_floats_after_name(data_view, stats_local_viterbi_mu, stats_local_viterbi_lambda);
                break;
            case 'F': // Forward
                parse_two_floats_after_name(data_view, stats_local_forward_theta, stats_local_forward_lambda);
                break;
        }
    }
}

void Hmm::extract_probabilities(std::ifstream& file) {
    // expecting COMPO tag
    auto data = std::string {};
    data = read_value_after_tag(file, "COMPO");

    insert_emissions.reserve(model_length + 1);
    match_emissions.reserve(model_length + 1);
    transitions.reserve(model_length + 1);

    // skip COMPO, next line is insert_emissions[0]
    std::getline(file, data);
    insert_emissions.push_back(parse_probabilities<NUM_OF_AMINO_ACIDS>(data));
    std::getline(file, data);
    transitions.push_back(parse_probabilities<NUM_OF_TRANSITIONS>(data));
    // match_emissions[0] should be not filled according to the logic of HMM.
    match_emissions.push_back(Probabilities_array<NUM_OF_AMINO_ACIDS>());

    // parse nodes 1..model_length
    for (size_t i = 1; i <= model_length; ++i) {
        data = read_value_after_tag(file, std::to_string(i));
        match_emissions.push_back(parse_probabilities<NUM_OF_AMINO_ACIDS>(data));
        std::getline(file, data);
        insert_emissions.push_back(parse_probabilities<NUM_OF_AMINO_ACIDS>(data));
        std::getline(file, data);
        transitions.push_back(parse_probabilities<NUM_OF_TRANSITIONS>(data));
    }
}
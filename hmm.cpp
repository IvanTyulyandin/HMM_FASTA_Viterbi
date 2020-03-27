#include "hmm.h"

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
}


Hmm::Hmm(const std::string& file_path) {
    auto file = std::ifstream(file_path);
    if (file.fail()) {
        std::cout << "Failed to open " << file_path << '\n';
        return;
    }
    extract_length(file);
    extract_stats_local(file);

    file.close();
}


void Hmm::extract_length(std::ifstream& file) {
    length = std::stoi(read_value_after_tag(file, "LENG"));
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
            case 'V': // Viterbi
                parse_two_floats_after_name(data_view, stats_local_viterbi_mu, stats_local_viterbi_lambda);
            case 'F': // Forward
                parse_two_floats_after_name(data_view, stats_local_forward_theta, stats_local_forward_lambda);
        }
    }
}

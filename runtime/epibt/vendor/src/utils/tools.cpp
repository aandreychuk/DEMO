#include <utils/tools.hpp>

std::vector<std::string> split(const std::string &str, char separator) {
    std::vector<std::string> result(1);
    for (char c: str) {
        if (c == separator) {
            if (!result.back().empty()) {
                result.emplace_back("");
            }
        } else {
            result.back() += c;
        }
    }
    if (result.back().empty()) {
        result.pop_back();
    }
    return result;
}

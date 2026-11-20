#include <utils/assert.hpp>

#include <iostream>
#include <unistd.h>

bool my_assert_failed(const std::string &message, const std::string &filename, const int line) {
    std::cerr.flush();
    std::cout.flush();
    std::cerr << "assert failed at " << filename << ":" << line << '\n';
    std::cerr << "message: \"" << message << "\"\n";
    std::cerr.flush();
    std::cout.flush();
    _exit(42);
    return true;
}

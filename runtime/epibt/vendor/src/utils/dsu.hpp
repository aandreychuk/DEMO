#pragma once

#include <cstdint>
#include <vector>

class DSU {
    std::vector<uint32_t> parent;
    std::vector<uint32_t> size;

public:
    DSU() = default;

    explicit DSU(uint32_t n);

    uint32_t get(uint32_t v);

    uint32_t get_size(uint32_t v);

    void uni(uint32_t a, uint32_t b);
};

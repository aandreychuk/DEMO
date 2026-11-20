#pragma once

#include <cstdint>
#include <iostream>
#include <vector>

// contains information about map:
// rows, cols, is_free
class Map {
    uint32_t rows = 0;
    uint32_t cols = 0;

    std::vector<bool> map;

public:
    Map() = default;

    [[nodiscard]] uint32_t get_rows() const;

    [[nodiscard]] uint32_t get_cols() const;

    [[nodiscard]] uint32_t get_size() const;

    [[nodiscard]] bool is_free(uint32_t pos) const;

    friend std::istream &operator>>(std::istream &input, Map &map);
};

Map &get_map();

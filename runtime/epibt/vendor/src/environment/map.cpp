#include <environment/map.hpp>

#include <utils/assert.hpp>
#include <utils/dsu.hpp>

uint32_t Map::get_rows() const {
    return rows;
}

uint32_t Map::get_cols() const {
    return cols;
}

uint32_t Map::get_size() const {
    return rows * cols + 1;
}

bool Map::is_free(uint32_t pos) const {
    ASSERT(pos < map.size(), "invalid pos");
    return map[pos];
}

std::istream &operator>>(std::istream &input, Map &map) {
    ASSERT(input, "unable to read");
    std::string line;
    std::getline(input, line);
    ASSERT(line == "type octile", "unexpected map file format");
    input >> line;
    ASSERT(line == "height", "unexpected map file format");
    input >> map.rows;
    input >> line;
    ASSERT(line == "width", "unexpected map file format");
    input >> map.cols;
    input >> line;
    ASSERT(line == "map", "unexpected map file format");
    map.map.assign(map.rows * map.cols + 1, false);
    for (uint32_t row = 0; row < map.rows; row++) {
        input >> line;
        ASSERT(line.size() == map.cols, "unexpected map file format");
        for (uint32_t col = 0; col < map.cols; col++) {
            ASSERT(line[col] == '.' || line[col] == '@' || line[col] == 'T', "unexpected cell type");
            map.map[row * map.cols + col + 1] = (line[col] == '.');
        }
    }
    return input;
}

Map &get_map() {
    static Map map;
    return map;
}

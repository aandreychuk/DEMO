#pragma once

#include <environment/position.hpp>

#include <istream>
#include <vector>

struct Robot {
    uint32_t node = 0;
    Position pos;
    uint32_t task_id = -1;
    uint32_t target = 0;
    uint32_t priority = -1;// -1 if disabled agent

    [[nodiscard]] bool is_disable() const;
};

class Robots {
    std::vector<Robot> robots;

public:
    Robots() = default;

    explicit Robots(uint32_t robots_num);

    Robot &operator[](uint32_t index);

    const Robot &operator[](uint32_t index) const;

    [[nodiscard]] uint32_t size() const;

    friend std::istream &operator>>(std::istream &input, Robots &robots);
};

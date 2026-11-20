#include <environment/robot.hpp>

#include <environment/graph.hpp>
#include <utils/assert.hpp>
#include <utils/tools.hpp>

bool Robot::is_disable() const {
    return priority == -1;
}

Robots::Robots(uint32_t robots_num) : robots(robots_num) {
}

Robot &Robots::operator[](uint32_t index) {
    ASSERT(index < robots.size(), "invalid index");
    return robots[index];
}

const Robot &Robots::operator[](uint32_t index) const {
    ASSERT(index < robots.size(), "invalid index");
    return robots[index];
}

uint32_t Robots::size() const {
    return robots.size();
}

std::istream &operator>>(std::istream &input, Robots &robots) {
    ASSERT(input, "invalid input");
    std::string line;
    std::getline(input, line);
    ASSERT(line == "agent id,row,col", "invalid header");
    uint32_t expected_id = 0;
    while (std::getline(input, line)) {
        auto strs = split(line, ',');
        ASSERT(strs.size() == 3, "invalid csv line");
        uint32_t id = std::stoll(strs[0]);
        uint32_t row = std::stoll(strs[1]);
        uint32_t col = std::stoll(strs[2]);
        ASSERT(id == expected_id, "invalid id");

        Robot robot;
#ifdef ENABLE_ROTATE_MODEL
        robot.pos = Position(row, col, 0);
#else
        robot.pos = Position(row, col);
#endif
        robot.node = get_graph().get_node(robot.pos);
        robots.robots.push_back(robot);
        expected_id++;
    }
    return input;
}

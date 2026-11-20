#include <environment/heuristic_matrix.hpp>
#include <planner/epibt/epibt.hpp>
#include <planner/epibt/operations_map.hpp>

#include <sstream>
#include <vector>

namespace {
std::vector<uint32_t> previous_operations;
constexpr int input_stride = 4;
constexpr int output_stride = 3 + EPIBT_DEPTH_VALUE;
}

extern "C" {

// The browser uses zero-based cells. Upstream reserves cell zero as a sentinel.
void epibt_init(int width, int height, const char *cells) {
    std::stringstream map_text;
    map_text << "type octile\nheight " << height << "\nwidth " << width << "\nmap\n";
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            map_text << (cells[y * width + x] == '#' ? '@' : '.');
        }
        map_text << '\n';
    }
    map_text >> get_map();
    const GraphGuidance guidance(height, width);
    get_graph() = Graph(get_map(), guidance);
    get_hm() = HeuristicMatrix(get_graph());
    if (get_operations().empty()) init_operations();
    get_omap() = OperationsMap(get_graph(), get_operations());
    previous_operations.clear();
}

void epibt_step(int count, const int32_t *input, int32_t *output) {
    Robots robots(count);
    if (previous_operations.size() != static_cast<size_t>(count)) {
        previous_operations.assign(count, 0);
    }

    for (int id = 0; id < count; ++id) {
        const auto *state = input + id * input_stride;
        auto &robot = robots[id];
        robot.pos = Position(state[0] + 1, state[1]);
        robot.node = get_graph().get_node(robot.pos);
        robot.target = state[2] + 1;
        robot.priority = get_hm().get(robot.node, robot.target);
        if (state[3] != state[2]) {
            // Match upstream TestSystem's priority for a task with two stops.
            const auto pickup_node = get_graph().get_node(Position(robot.target, 0));
            robot.priority += get_hm().get(pickup_node, state[3] + 1);
        }
    }

    EPIBT planner(robots, get_now() + Milliseconds(30), previous_operations);
    planner.solve();
    const auto operations = planner.get_desires();

    for (int id = 0; id < count; ++id) {
        const auto &operation = get_operations()[operations[id]];
        const auto next = robots[id].pos.simulate(operation.front());
        auto *state = output + id * output_stride;
        state[0] = next.get_pos() - 1;
        state[1] = next.get_dir();
        state[2] = robots[id].priority;
        for (uint32_t depth = 0; depth < EPIBT_DEPTH_VALUE; ++depth) {
            state[3 + depth] = static_cast<int32_t>(operation[depth]);
        }
        previous_operations[id] = get_operation_next(operations[id]);
    }
}

}

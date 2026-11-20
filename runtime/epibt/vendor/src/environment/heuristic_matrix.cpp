#include <environment/heuristic_matrix.hpp>

#include <utils/assert.hpp>
#include <utils/tools.hpp>

#include <queue>

void HeuristicMatrix::build(uint32_t source, const Graph &graph) {
    auto &dists = matrix[source];
    dists.assign(graph.get_nodes_size(), -1);

    // (dist, node)
    std::priority_queue<std::pair<uint32_t, uint32_t>, std::vector<std::pair<uint32_t, uint32_t>>, std::greater<>> heap;

    std::vector<bool> visited(graph.get_nodes_size());

#ifdef ENABLE_ROTATE_MODEL
    for (uint32_t dir = 0; dir < DIRECTIONS_NUM; dir++) {
        heap.push({0, graph.get_node(Position(source, dir))});
    }
#else
    heap.push({0, graph.get_node(Position(source))});
#endif

    while (!heap.empty()) {
        auto [dist, node] = heap.top();
        heap.pop();

        if (visited[node]) {
            continue;
        }
        visited[node] = true;
        dists[node] = dist;

        for (uint32_t action = 0; action < ACTIONS_NUM - 1; action++) {
            uint32_t to = 0;
#ifdef ENABLE_ROTATE_MODEL
            ASSERT(static_cast<ActionType>(action) == ActionType::FORWARD ||
                           static_cast<ActionType>(action) == ActionType::ROTATE ||
                           static_cast<ActionType>(action) == ActionType::COUNTER_ROTATE,
                   "invalid action");

            if (action == 0) {
                to = get_graph().get_to_node(node, 1);
                to = get_graph().get_to_node(to, 1);
                to = get_graph().get_to_node(to, 0);
                if (to) {
                    to = get_graph().get_to_node(to, 1);
                    to = get_graph().get_to_node(to, 1);
                }
            } else if (action == 1) {
                to = get_graph().get_to_node(node, 2);
            } else if (action == 2) {
                to = get_graph().get_to_node(node, 1);
            }
#else
            ASSERT(static_cast<ActionType>(action) == ActionType::EAST ||         //
                           static_cast<ActionType>(action) == ActionType::SOUTH ||//
                           static_cast<ActionType>(action) == ActionType::WEST || //
                           static_cast<ActionType>(action) == ActionType::NORTH,
                   "invalid action");
            /*
            0 EAST >
            1 SOUTH v
            2 WEST <
            3 NORTH ^

            0 -> 2
            1 -> 3
            2 -> 0
            3 -> 1
            (action + 2) % 4
             */
            to = get_graph().get_to_node(node, (action + 2) % 4);
#endif

            if (!to) {
                continue;
            }

            ASSERT(get_graph().get_to_node(to, action) == node, "invalid to");

            if (!visited[to]) {
                uint64_t to_dist = dist + graph.get_weight(to, action);
                ASSERT(to_dist == static_cast<uint16_t>(to_dist), "dist overflow");
                if (dists[to] > to_dist) {
                    dists[to] = to_dist;
                    heap.push({to_dist, to});
                }
            }
        }
    }
}

HeuristicMatrix::HeuristicMatrix(const Graph &graph) {
    matrix.resize(get_map().get_size());

    launch_threads(THREADS_NUM_VALUE, [&](uint32_t thr) {
        for (uint32_t pos = thr + 1; pos < matrix.size(); pos += THREADS_NUM_VALUE) {
            if (Position(pos
#ifdef ENABLE_ROTATE_MODEL
                         ,
                         0
#endif
                         )
                        .is_valid()) {
                build(pos, graph);
            }
        }
    });
}

uint32_t HeuristicMatrix::get(uint32_t source, uint32_t target) const {
    ASSERT(get_map().get_size() == matrix.size(), "unmatch sizes");
    ASSERT(target && target < matrix.size(), "invalid target");
    ASSERT(source && source < matrix[target].size(), "invalid source");
    return matrix[target][source];
}

HeuristicMatrix &get_hm() {
    static HeuristicMatrix hm;
    return hm;
}

#include <environment/graph.hpp>

#include <utils/assert.hpp>

#include <map>

Graph::Graph(const Map &map, const GraphGuidance &gg) {
    pos_to_node.resize(map.get_size());
    node_to_pos.resize(1);
    for (uint32_t pos = 1; pos < map.get_size(); pos++) {
        if (!map.is_free(pos)) {
            continue;
        }
#ifdef ENABLE_ROTATE_MODEL
        for (uint32_t dir = 0; dir < DIRECTIONS_NUM; dir++) {
            Position p(pos, dir);
            ASSERT(p.is_valid(), "p is invalid");
            pos_to_node[pos][dir] = node_to_pos.size();
            node_to_pos.push_back(p);
        }
#else
        Position p(pos);
        ASSERT(p.is_valid(), "p is invalid");
        pos_to_node[pos] = node_to_pos.size();
        node_to_pos.push_back(p);
#endif
    }

    to_node.resize(node_to_pos.size());
    to_edge.resize(node_to_pos.size());
    weight.resize(node_to_pos.size());

    std::map<std::pair<uint32_t, uint32_t>, uint32_t> edges;
    for (uint32_t node = 1; node < node_to_pos.size(); node++) {
        for (uint32_t action = 0; action < ACTIONS_NUM; action++) {
            Position p = node_to_pos[node];
            Position to = p.simulate(static_cast<ActionType>(action));
            if (!to.is_valid()) {
                continue;
            }

            to_node[node][action] = get_node(to);
            weight[node][action] = gg.get(p, action);

            uint32_t a = p.get_pos();
            uint32_t b = to.get_pos();

            if (a == b) {
                continue;
            }

            if (a > b) {
                std::swap(a, b);
            }
            if (!edges.count({a, b})) {
                edges[{a, b}] = edges.size() + 1;
            }

            to_edge[node][action] = edges[{a, b}];
        }
    }
    edges_size = edges.size() + 1;
}

uint32_t Graph::get_nodes_size() const {
    return node_to_pos.size();
}

uint32_t Graph::get_edges_size() const {
    return edges_size;
}

Position Graph::get_pos(uint32_t node) const {
    ASSERT(0 < node && node < node_to_pos.size(), "invalid node: " + std::to_string(node));
    return node_to_pos[node];
}

uint32_t Graph::get_node(const Position &pos) const {
    ASSERT(pos.get_pos() < pos_to_node.size(), "invalid pos");
    ASSERT(pos.is_valid(), "invalid pos");

#ifdef ENABLE_ROTATE_MODEL
    ASSERT(pos.get_dir() < DIRECTIONS_NUM, "invalid dir");
    return pos_to_node[pos.get_pos()][pos.get_dir()];
#else
    return pos_to_node[pos.get_pos()];
#endif
}

uint32_t Graph::get_node(uint32_t pos
#ifdef ENABLE_ROTATE_MODEL
        , uint32_t dir
#endif
) const {
    ASSERT(pos < pos_to_node.size(), "invalid pos");
#ifdef ENABLE_ROTATE_MODEL
    ASSERT(dir < DIRECTIONS_NUM, "invalid dir");
    return pos_to_node[pos][dir];
#else
    return pos_to_node[pos];
#endif
}

uint32_t Graph::get_to_node(uint32_t node, uint32_t action) const {
    ASSERT(0 < node && node < to_node.size(), "invalid node");
    ASSERT(action < ACTIONS_NUM, "invalid action");
    return to_node[node][action];
}

uint32_t Graph::get_to_edge(uint32_t node, uint32_t action) const {
    ASSERT(0 < node && node < to_edge.size(), "invalid node");
    ASSERT(action < ACTIONS_NUM, "invalid action");
    return to_edge[node][action];
}

uint32_t Graph::get_weight(uint32_t node, uint32_t action) const {
    ASSERT(0 < node && node < to_edge.size(), "invalid node");
    ASSERT(action < ACTIONS_NUM, "invalid action");
    return weight[node][action];
}

Graph &get_graph() {
    static Graph graph;
    return graph;
}

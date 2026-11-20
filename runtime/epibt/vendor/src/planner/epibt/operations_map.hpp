#pragma once

#include <environment/graph.hpp>
#include <planner/epibt/operations.hpp>

#include <tuple>

class OperationsMap {

    // map_nodes[node][operation] = nodes path
    std::vector<std::vector<Path>> map_nodes;

    // map_edges[node][operation] = edges path
    std::vector<std::vector<Path>> map_edges;

    // map_poses[node][operation] = poses path
    std::vector<std::vector<Path>> map_poses;

    void build(uint32_t source, const std::vector<Operation> &operations);

public:
    OperationsMap() = default;

    OperationsMap(const Graph &graph, const std::vector<Operation> &operations);

    [[nodiscard]] const Path &get_nodes_path(uint32_t node, uint32_t operation) const;

    [[nodiscard]] const Path &get_edges_path(uint32_t node, uint32_t operation) const;

    [[nodiscard]] const Path &get_poses_path(uint32_t node, uint32_t operation) const;
};

OperationsMap &get_omap();

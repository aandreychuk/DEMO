/*
 * distance table with lazy evaluation, using BFS
 */
#pragma once

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

struct DistTable {
  const int K;  // number of vertices
  std::vector<std::vector<int>>
      table;  // distance table to goals, index: agent-id & vertex-id
  std::vector<std::vector<int>>
      table_to_starts;  // distance table to starts, index: agent-id & vertex-id
  std::vector<std::queue<Vertex *>> OPEN;  // search queue

  int get(const int i, const int v_id);   // agent, vertex-id (distance to goal)
  int get(const int i, const Vertex *v);  // agent, vertex (distance to goal)

  int get_to_start(const int i, const int v_id) const;   // agent, vertex-id (distance to start)
  int get_to_start(const int i, const Vertex *v) const;  // agent, vertex (distance to start)

  DistTable(const Instance &ins);
  DistTable(const Instance *ins);

  void setup(const Instance *ins);  // initialization
  void set_goal(const int i, Vertex *goal,
                const std::vector<char> *blocked = nullptr);
};

#include "../include/dist_table.hpp"

DistTable::DistTable(const Instance &ins)
    : K(ins.G->V.size()),
      table(ins.N, std::vector<int>(K, K)),
      table_to_starts(ins.N, std::vector<int>(K, K))
{
  setup(&ins);
}

DistTable::DistTable(const Instance *ins)
    : K(ins->G->V.size()),
      table(ins->N, std::vector<int>(K, K)),
      table_to_starts(ins->N, std::vector<int>(K, K))
{
  setup(ins);
}

void DistTable::setup(const Instance *ins)
{
  auto bfs_to_goal = [&](const int i) {
    set_goal(i, ins->goals[i]);
  };

  auto bfs_to_start = [&](const int i) {
    auto s_i = ins->starts[i];
    auto Q = std::queue<Vertex *>({s_i});
    table_to_starts[i][s_i->id] = 0;
    while (!Q.empty()) {
      auto n = Q.front();
      Q.pop();
      const int d_n = table_to_starts[i][n->id];
      for (auto &m : n->neighbor) {
        const int d_m = table_to_starts[i][m->id];
        if (d_n + 1 >= d_m) continue;
        table_to_starts[i][m->id] = d_n + 1;
        Q.push(m);
      }
    }
  };

  auto pool = std::vector<std::future<void>>();
  for (size_t i = 0; i < ins->N; ++i) {
    pool.emplace_back(std::async(std::launch::async, bfs_to_goal, i));
    pool.emplace_back(std::async(std::launch::async, bfs_to_start, i));
  }
}

void DistTable::set_goal(const int i, Vertex *goal,
                         const std::vector<char> *blocked)
{
  if (i < 0 || i >= static_cast<int>(table.size()) || goal == nullptr) return;
  auto &distances = table[i];
  std::fill(distances.begin(), distances.end(), K);
  if (blocked != nullptr && goal->id < static_cast<int>(blocked->size()) &&
      (*blocked)[goal->id]) {
    return;
  }
  auto queue = std::queue<Vertex *>({goal});
  distances[goal->id] = 0;
  while (!queue.empty()) {
    auto *vertex = queue.front();
    queue.pop();
    const int distance = distances[vertex->id];
    for (auto *neighbor : vertex->neighbor) {
      if (blocked != nullptr &&
          neighbor->id < static_cast<int>(blocked->size()) &&
          (*blocked)[neighbor->id]) {
        continue;
      }
      if (distance + 1 >= distances[neighbor->id]) continue;
      distances[neighbor->id] = distance + 1;
      queue.push(neighbor);
    }
  }
}

int DistTable::get(const int i, const int v_id) { return table[i][v_id]; }

int DistTable::get(const int i, const Vertex *v) { return get(i, v->id); }

int DistTable::get_to_start(const int i, const int v_id) const { return table_to_starts[i][v_id]; }

int DistTable::get_to_start(const int i, const Vertex *v) const { return get_to_start(i, v->id); }

#include "../include/collision_table.hpp"

#include <algorithm>
#include <climits>

#include "../include/dist_table.hpp"

CollisionTable::CollisionTable(const Instance *ins, bool _no_use_collision_cnt)
    : body(ins->G->size()),
      body_last(ins->G->size()),
      body_last_agent(ins->G->size()),
      collision_cnt(0),
      N(ins->N),
      no_use_collision_cnt(_no_use_collision_cnt),
      goal_block_time(ins->G->size(), INT_MAX),
      vertex_cells_count(ins->G->size(), 0),
      active_vertex_flag(ins->G->size(), 0),
      dirty_vertex_flag(ins->G->size(), 0)
{
  active_vertices.reserve(ins->G->size() / 8 + 16);
  dirty_vertices.reserve(ins->G->size() / 8 + 16);
}

CollisionTable::~CollisionTable() {}

int CollisionTable::getCollisionCost(const Vertex *v_from, const Vertex *v_to,
                                     const int t_from)
{
  const int t_to = t_from + 1;
  auto collision = 0;
  // vertex collision
  if (t_to < body[v_to->id].size()) {
    collision += body[v_to->id][t_to].size();
  }
  // edge collision
  if (t_to < body[v_from->id].size() && t_from < body[v_to->id].size()) {
    for (auto j : body[v_from->id][t_to]) {
      for (auto k : body[v_to->id][t_from]) {
        if (j == k) ++collision;
      }
    }
  }
  // goal collision
  for (auto last_timestep : body_last[v_to->id]) {
    if (t_to > last_timestep) ++collision;
  }
  return collision;
}

void CollisionTable::enrollPath(const int i, Path &path)
{
  if (path.empty()) return;
  ++mutation_version;
  if (goal_penalty_enabled && i >= 0 &&
      i < static_cast<int>(is_path_unbuilt.size())) {
    is_path_unbuilt[i] = 0;
  }
  const auto T_i = path.size() - 1;
  for (auto t = 0; t <= T_i; ++t) {
    auto v = path[t];

    // update collision count
    if (t > 0 && !no_use_collision_cnt) {
      collision_cnt += getCollisionCost(path[t - 1], path[t], t - 1);
    }

    // mark this vertex dirty for incremental snapshot rebuilds
    mark_dirty_vertex(v->id);

    // register
    while (body[v->id].size() <= t) body[v->id].emplace_back();
    auto &cell = body[v->id][t];
    const bool was_empty = cell.empty();
    cell.push_back(i);
    ++total_agent_refs;
    if (was_empty) {
      ++total_occupied_cells;
      const int vid = v->id;
      if (++vertex_cells_count[vid] == 1) {
        // 0 -> 1 transition: vertex becomes active.
        active_vertex_flag[vid] = 1;
        // Insert keeping the list sorted; common case is append near the end
        // because paths share chokepoints whose ids cluster spatially.
        auto it = std::lower_bound(active_vertices.begin(),
                                   active_vertices.end(), vid);
        active_vertices.insert(it, vid);
      }
    }
  }

  // goal
  const int goal_v = path.back()->id;
  body_last[goal_v].push_back(static_cast<int>(T_i));
  body_last_agent[goal_v].emplace_back(i, static_cast<int>(T_i));
  if (static_cast<int>(T_i) < goal_block_time[goal_v]) {
    goal_block_time[goal_v] = static_cast<int>(T_i);
  }
  if (!no_use_collision_cnt) {
    auto &&entry = body[goal_v];
    for (auto t = T_i + 1; t < entry.size(); ++t) {
      collision_cnt += entry[t].size();
    }
  }

  // Phase E: mirror this enroll onto the CUDA-SIPP workspace's device buffers.
  if (path_listener_ != nullptr) {
    path_listener_(path_listener_userdata_, i, path, /*is_enroll=*/true,
                   /*is_prefix=*/false);
  }
}

void CollisionTable::clearPath(const int i, Path &path)
{
  if (path.empty()) return;
  if (path[0] == nullptr) return;
  ++mutation_version;
  if (goal_penalty_enabled && i >= 0 &&
      i < static_cast<int>(is_path_unbuilt.size())) {
    is_path_unbuilt[i] = 1;
  }
  const auto T_i = (int)path.size() - 1;
  for (auto t = 0; t <= T_i; ++t) {
    auto v = path[t];
    mark_dirty_vertex(v->id);
    auto &&entry = body[v->id][t];

    // remove entry
    for (auto itr = entry.begin(); itr != entry.end();) {
      if (*itr == i) {
        entry.erase(itr);
        --total_agent_refs;
        if (entry.empty()) {
          --total_occupied_cells;
          const int vid = v->id;
          if (--vertex_cells_count[vid] == 0) {
            // 1 -> 0 transition: vertex becomes inactive; drop from sorted list.
            active_vertex_flag[vid] = 0;
            auto it = std::lower_bound(active_vertices.begin(),
                                       active_vertices.end(), vid);
            if (it != active_vertices.end() && *it == vid) {
              active_vertices.erase(it);
            }
          }
        }
        break;
      } else {
        ++itr;
      }
    }

    // update collision count
    if (t > 0 && !no_use_collision_cnt) {
      collision_cnt -= getCollisionCost(path[t - 1], path[t], t - 1);
    }
  }

  // goal
  const int goal_v = path.back()->id;
  auto &&entry_body_last = body_last[goal_v];
  bool removed_T_i = false;
  for (auto itr = entry_body_last.begin(); itr != entry_body_last.end();) {
    if (*itr == T_i) {
      entry_body_last.erase(itr);
      removed_T_i = true;
      break;
    } else {
      ++itr;
    }
  }
  if (goal_v >= 0 && goal_v < static_cast<int>(body_last_agent.size())) {
    auto &entry_body_last_agent = body_last_agent[goal_v];
    for (auto itr = entry_body_last_agent.begin(); itr != entry_body_last_agent.end();) {
      if (itr->first == i && itr->second == T_i) {
        entry_body_last_agent.erase(itr);
        break;
      } else {
        ++itr;
      }
    }
  }
  if (removed_T_i) {
    if (entry_body_last.empty()) {
      goal_block_time[goal_v] = INT_MAX;
    } else {
      goal_block_time[goal_v] =
          *std::min_element(entry_body_last.begin(), entry_body_last.end());
    }
  }
  if (!no_use_collision_cnt) {
    auto &&entry_body = body[goal_v];
    for (auto t = T_i + 1; t < entry_body.size(); ++t) {
      collision_cnt -= entry_body[t].size();
    }
  }

  // Phase E: mirror this clear onto the CUDA-SIPP workspace's device buffers.
  if (path_listener_ != nullptr) {
    path_listener_(path_listener_userdata_, i, path, /*is_enroll=*/false,
                   /*is_prefix=*/false);
  }
}

void CollisionTable::enrollPrefix(const int i, Path &prefix)
{
  if (prefix.empty()) return;
  ++mutation_version;
  const auto T_i = prefix.size() - 1;
  for (size_t t = 0; t <= T_i; ++t) {
    auto v = prefix[t];
    if (t > 0 && !no_use_collision_cnt) {
      collision_cnt += getCollisionCost(prefix[t - 1], prefix[t], static_cast<int>(t - 1));
    }
    mark_dirty_vertex(v->id);
    while (body[v->id].size() <= t) body[v->id].emplace_back();
    auto &cell = body[v->id][t];
    const bool was_empty = cell.empty();
    cell.push_back(i);
    ++total_agent_refs;
    if (was_empty) {
      ++total_occupied_cells;
      const int vid = v->id;
      if (++vertex_cells_count[vid] == 1) {
        active_vertex_flag[vid] = 1;
        auto it = std::lower_bound(active_vertices.begin(),
                                   active_vertices.end(), vid);
        active_vertices.insert(it, vid);
      }
    }
  }
  // Deliberately DO NOT touch body_last / goal_block_time — prefix.back() is
  // not the agent's real goal. Deliberately DO NOT touch is_path_unbuilt —
  // the agent's actual goal has not yet been reached, so goal-penalty on
  // his goal vertex should still apply to other agents.

  // Phase E: mirror this prefix-enroll onto the CUDA-SIPP workspace.
  if (path_listener_ != nullptr) {
    path_listener_(path_listener_userdata_, i, prefix, /*is_enroll=*/true,
                   /*is_prefix=*/true);
  }
}

void CollisionTable::clearPrefix(const int i, Path &prefix)
{
  if (prefix.empty()) return;
  if (prefix[0] == nullptr) return;
  ++mutation_version;
  const auto T_i = static_cast<int>(prefix.size()) - 1;
  for (auto t = 0; t <= T_i; ++t) {
    auto v = prefix[t];
    mark_dirty_vertex(v->id);
    auto &&entry = body[v->id][t];
    for (auto itr = entry.begin(); itr != entry.end();) {
      if (*itr == i) {
        entry.erase(itr);
        --total_agent_refs;
        if (entry.empty()) {
          --total_occupied_cells;
          const int vid = v->id;
          if (--vertex_cells_count[vid] == 0) {
            active_vertex_flag[vid] = 0;
            auto it = std::lower_bound(active_vertices.begin(),
                                       active_vertices.end(), vid);
            if (it != active_vertices.end() && *it == vid) {
              active_vertices.erase(it);
            }
          }
        }
        break;
      } else {
        ++itr;
      }
    }
    if (t > 0 && !no_use_collision_cnt) {
      collision_cnt -= getCollisionCost(prefix[t - 1], prefix[t], t - 1);
    }
  }
  // Mirror of enrollPrefix — skip body_last and is_path_unbuilt updates.

  // Phase E: mirror this prefix-clear onto the CUDA-SIPP workspace.
  if (path_listener_ != nullptr) {
    path_listener_(path_listener_userdata_, i, prefix, /*is_enroll=*/false,
                   /*is_prefix=*/true);
  }
}

void CollisionTable::enable_goal_penalty(const Instance *ins, DistTable *D)
{
  goal_penalty_enabled = true;
  goal_thresholds_by_vertex.assign(ins->G->size(), {});
  is_path_unbuilt.assign(ins->N, 1);
  for (int a = 0; a < ins->N; ++a) {
    const int goal_id = ins->goals[a]->id;
    const int t_star = D->get(a, ins->starts[a]);
    goal_thresholds_by_vertex[goal_id].emplace_back(a, t_star);
  }
}

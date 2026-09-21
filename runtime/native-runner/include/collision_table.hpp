/*
 * fast collision checking, used in SUO and refinner
 */
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

struct DistTable;

struct CollisionTable {
  // vertex, time, agents
  std::vector<std::vector<std::vector<int>>> body;
  std::vector<std::vector<int>> body_last;
  // Owner-aware goal holds. body_last keeps the historical time-only API;
  // this mirror lets masked SIPP ignore goal-hold constraints by agent id
  // without physically removing paths from CT/SI.
  std::vector<std::vector<std::pair<int, int>>> body_last_agent;  // (agent, arrival_t)
  int collision_cnt;
  const int N;
  const bool no_use_collision_cnt;
  // Bumped on every enrollPath/clearPath so CUDA-SIPP workspace can detect
  // when host-side snapshot / safe-interval caches become stale.
  std::uint64_t mutation_version = 0;

  // Incrementally-maintained accelerators for `build_sipp_gpu_collision_snapshot`.
  //   total_occupied_cells — # of (v,t) cells with at least one agent.
  //   total_agent_refs     — sum over all (v,t) of body[v][t].size().
  //   goal_block_time[v]   — min T_i among paths ending at v, or INT_MAX.
  //   vertex_cells_count[v] — # of non-empty body[v][t] cells at vertex v;
  //                           used to skip empty vertices in CSR build.
  //   active_vertex_flag[v] — 1 iff vertex_cells_count[v] > 0.
  //   active_vertices      — sorted list of vertices with at least one cell.
  //                          Maintained as a sorted vector so the snapshot
  //                          builder can iterate in vertex-id order without
  //                          a pre-sort pass.
  std::vector<int> goal_block_time;
  int total_occupied_cells = 0;
  int total_agent_refs = 0;
  std::vector<int> vertex_cells_count;
  std::vector<std::uint8_t> active_vertex_flag;
  std::vector<int> active_vertices;

  // Dirty-vertex tracking for incremental snapshot rebuilds. Every enrollPath
  // / clearPath call marks all vertices touched by the path. Consumers of
  // the snapshot patch only these vertices' slabs and then call
  // clear_dirty_vertices() to reset the tracking. The flag array is
  // deduplicated so pushing the same vertex twice inside one enrollPath is
  // cheap.
  mutable std::vector<int> dirty_vertices;
  mutable std::vector<std::uint8_t> dirty_vertex_flag;
  // Monotonically increasing; bumps each time clear_dirty_vertices() is
  // called. Lets consumers detect whether they have seen the current
  // cleared-state (post-patch) or need a full rebuild.
  mutable std::uint64_t dirty_epoch = 0;

  void mark_dirty_vertex(int vid) const {
    if (vid < 0 || vid >= static_cast<int>(dirty_vertex_flag.size())) return;
    if (!dirty_vertex_flag[vid]) {
      dirty_vertex_flag[vid] = 1;
      dirty_vertices.push_back(vid);
    }
  }
  void clear_dirty_vertices() const {
    for (int v : dirty_vertices) dirty_vertex_flag[v] = 0;
    dirty_vertices.clear();
    ++dirty_epoch;
  }

  // Goal-blocking soft penalty (enabled per LNS session): if agent B traverses
  // a vertex v that is the goal of agent A, and B arrives at v at time t >
  // t_A* (A's shortest arrival time), B's path cost gets extra (t - t_A*).
  // Penalty only fires for agents A whose path is currently unbuilt (entry
  // in is_path_unbuilt is 1) — once A is enrolled, hard collisions via CT
  // already cover the blockage.
  bool goal_penalty_enabled = false;
  std::vector<std::vector<std::pair<int, int>>> goal_thresholds_by_vertex;  // (owner, t_star)
  std::vector<std::uint8_t> is_path_unbuilt;  // 1 = path currently cleared, 0 = enrolled
  bool collect_sipp_blocker_events = true;

  CollisionTable(const Instance *ins, bool _no_use_collision_cnt = false);
  ~CollisionTable();

  int getCollisionCost(const Vertex *v_from, const Vertex *v_to,
                       const int t_from);
  void enrollPath(const int i, Path &path);
  void clearPath(const int i, Path &path);
  // Enroll only a prefix of a path — used for tail-destroy. Unlike
  // enrollPath, this does NOT mark prefix.back() as a permanent goal
  // occupancy (body_last / goal_block_time unchanged) and does NOT flip
  // is_path_unbuilt to 0 (the agent's goal has not actually been reached).
  void enrollPrefix(const int i, Path &prefix);
  // Inverse of enrollPrefix: removes the cells without touching goal state.
  void clearPrefix(const int i, Path &prefix);
  std::uint64_t version() const { return mutation_version; }

  // Precompute goal_thresholds_by_vertex from instance + DistTable. After
  // this call is_path_unbuilt starts as all-1; subsequent enrollPath/clearPath
  // keep it in sync. Safe to call before the initial enroll loop.
  void enable_goal_penalty(const Instance *ins, DistTable *D);

  // ---- Path-mutation listener (Phase E: elite-pool device mirror) ----
  //
  // Optional callback invoked AFTER each enrollPath / clearPath / enrollPrefix
  // / clearPrefix mutation completes. Lets the CUDA-SIPP workspace mirror
  // every CT mutation onto the device's per-worker buffers (d_paths_,
  // d_who_at_, ...) without re-walking CT body.
  //
  // The callback receives (userdata, agent_id, path, is_enroll, is_prefix).
  //   - is_enroll: true for enrollPath / enrollPrefix, false for clear*.
  //   - is_prefix: true for *Prefix variants (do not touch goal-blocker on
  //                the prefix's last vertex; it is not the agent's real goal).
  //   - path:      the path passed to the mutator. The vertex IDs in path
  //                are the cells whose body[v][t] entries just got
  //                appended/removed for this agent; mirroring them to
  //                who_at[v][t] keeps the device view consistent.
  //
  // Listener state is owned by the workspace; CT only stores the function
  // pointer + userdata. Clearing the listener (set_path_listener with
  // nullptr) is safe at any time.
  using PathListener = void (*)(void *userdata, int agent_id, const Path &path,
                                bool is_enroll, bool is_prefix);
  PathListener path_listener_ = nullptr;
  void *path_listener_userdata_ = nullptr;

  void set_path_listener(PathListener fn, void *ud) {
    path_listener_ = fn;
    path_listener_userdata_ = ud;
  }
  void clear_path_listener() {
    path_listener_ = nullptr;
    path_listener_userdata_ = nullptr;
  }
};

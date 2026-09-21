/*
 * implementation of PIBT
 *
 * references:
 * Priority Inheritance with Backtracking for Iterative Multi-agent Path
 * Finding. Keisuke Okumura, Manao Machida, Xavier Défago & Yasumasa Tamura.
 * Artificial Intelligence (AIJ). 2022.
 */
#pragma once
#include "dist_table.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "policy.hpp"
#include "utils.hpp"

struct PIBT {
  const Instance *ins;
  std::mt19937 MT;
  std::uniform_real_distribution<float> rrd;  // random, real distribution

  // solver utils
  const int N;  // number of agents
  const int V_size;
  DistTable *D;

  // specific to PIBT
  std::vector<int> occupied_now;   // for quick collision checking
  std::vector<int> occupied_next;  // for quick collision checking

  AgentPolicy policy;

  // recursion tracking for complexity estimation
  int recursion_count;       // total recursive calls in current expansion
  int max_recursion_depth;   // max recursion depth reached (per-agent)
  int current_depth;         // current recursion depth (tracking during funcPIBT)

  // Optional goal-departure diagnostics. Enabled with
  // LAGAT_PIBT_GOAL_DIAG=1 and otherwise inert.
  bool goal_diag_enabled;
  std::vector<int> chosen_rank;
  std::vector<char> displaced_on_entry;
  std::vector<char> action_prefilled;
  long long goal_departures = 0;
  long long goal_departures_top_leave = 0;
  long long goal_departures_top_leave_strict = 0;
  long long goal_departures_top_leave_tied_wait = 0;
  long long goal_departures_top_wait = 0;
  long long goal_departures_displaced = 0;
  long long goal_departures_prefilled = 0;
  long long goal_departures_non_top = 0;
  long long goal_arrivals = 0;

  // hyper parameters
  static bool SWAP;

  PIBT(const Instance *_ins, DistTable *_D, const PolicyConfig &config, int seed = 0);
  ~PIBT();

  bool set_new_config(const Config &Q_from, Config &Q_to,
                      const std::vector<int> &order,
                      const std::set<int> &default_policy_agents,
                      const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>> *action_history = nullptr,
                      const std::vector<std::vector<int>>
                          *forbidden_vertex_ids = nullptr);
  bool funcPIBT(const int i, const Config &Q_from, Config &Q_to,
                const std::set<int> &default_policy_agents,
                const std::vector<std::vector<int>>
                    *forbidden_vertex_ids = nullptr);

  // Get recursion metrics
  int get_recursion_count() const { return recursion_count; }
  int get_max_recursion_depth() const { return max_recursion_depth; }
  void reset_recursion_count() {
    recursion_count = 0;
    max_recursion_depth = 0;
    current_depth = 0;
  }
  int is_swap_required_and_possible(const int ai, const Config &Q_from,
                                    Config &Q_to, Vertex *v_i_target);
  bool is_swap_required(const int pusher, const int puller,
                        Vertex *v_pusher_origin, Vertex *v_puller_origin);
  bool is_swap_possible(Vertex *v_pusher_origin, Vertex *v_puller_origin);
};

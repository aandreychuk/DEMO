#include "../include/pibt.hpp"

#include <cmath>
#include <cstdlib>

bool PIBT::SWAP = true;

PIBT::PIBT(const Instance *_ins, DistTable *_D, const PolicyConfig &config, int seed)
    : ins(_ins),
      MT(seed),
      rrd(0, 1),
      N(ins->N),
      V_size(ins->G->size()),
      D(_D),
      occupied_now(V_size, NO_AGENT),
      occupied_next(V_size, NO_AGENT),
      policy(ins, D, config, seed),
      recursion_count(0),
      max_recursion_depth(0),
      current_depth(0),
      goal_diag_enabled([]() {
        const char *value = std::getenv("LAGAT_PIBT_GOAL_DIAG");
        return value != nullptr && value[0] != '\0' && value[0] != '0';
      }()),
      chosen_rank(N, -1),
      displaced_on_entry(N, false),
      action_prefilled(N, false)
{
}

PIBT::~PIBT()
{
  if (!goal_diag_enabled) return;
  Common::info(0, "[PIBT-GOAL-DIAG] policy=", policy.policy_type_,
               " departures=", goal_departures,
               " top_leave=", goal_departures_top_leave,
               " top_leave_strict=", goal_departures_top_leave_strict,
               " top_leave_tied_wait=", goal_departures_top_leave_tied_wait,
               " top_wait=", goal_departures_top_wait,
               " displaced=", goal_departures_displaced,
               " prefilled=", goal_departures_prefilled,
               " non_top=", goal_departures_non_top,
               " arrivals=", goal_arrivals);
}

bool PIBT::set_new_config(const Config &Q_from, Config &Q_to,
                          const std::vector<int> &order,
                          const std::set<int> &default_policy_agents,
                          const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>> *action_history,
                          const std::vector<std::vector<int>>
                              *forbidden_vertex_ids)
{
  bool success = true;
  // Reset recursion counters at the start of each expansion
  recursion_count = 0;
  max_recursion_depth = 0;
  current_depth = 0;
  if (goal_diag_enabled) {
    std::fill(chosen_rank.begin(), chosen_rank.end(), -1);
    std::fill(displaced_on_entry.begin(), displaced_on_entry.end(), false);
    for (int i = 0; i < N; ++i) {
      action_prefilled[i] = Q_to[i] != nullptr;
    }
  }

  // setup cache & constraints check
  for (auto i = 0; i < N; ++i) {
    // set occupied now
    occupied_now[Q_from[i]->id] = i;

    // set occupied next
    if (Q_to[i] != nullptr) {
      // vertex collision
      if (occupied_next[Q_to[i]->id] != NO_AGENT) {
        success = false;
        break;
      }
      // swap collision
      auto j = occupied_now[Q_to[i]->id];
      if (j != NO_AGENT && j != i && Q_to[j] == Q_from[i]) {
        success = false;
        break;
      }
      occupied_next[Q_to[i]->id] = i;
    }
  }

  if (success) {
    policy.set_preferences(Q_from, default_policy_agents, action_history);
    for (auto i : order) {
      if (Q_to[i] == nullptr &&
          !funcPIBT(i, Q_from, Q_to, default_policy_agents,
                    forbidden_vertex_ids)) {
        success = false;
        break;
      }
    }
  }

  if (success && goal_diag_enabled) {
    for (int i = 0; i < N; ++i) {
      const bool was_at_goal = Q_from[i] == ins->goals[i];
      const bool is_at_goal = Q_to[i] == ins->goals[i];
      if (!was_at_goal && is_at_goal) ++goal_arrivals;
      if (!was_at_goal || Q_to[i] == Q_from[i]) continue;

      ++goal_departures;
      if (action_prefilled[i]) ++goal_departures_prefilled;
      if (displaced_on_entry[i]) ++goal_departures_displaced;
      if (chosen_rank[i] > 0) ++goal_departures_non_top;

      const bool top_is_wait = policy.get(i, 0) == Q_from[i];
      if (top_is_wait) {
        ++goal_departures_top_wait;
      } else {
        ++goal_departures_top_leave;
        const auto &top_cost = std::get<1>(policy.preferences[i][0]);
        const auto wait_it = std::find_if(
            policy.preferences[i].begin(), policy.preferences[i].end(),
            [&](const auto &entry) {
              return std::get<0>(entry) == Q_from[i];
            });
        if (wait_it != policy.preferences[i].end()) {
          const auto &wait_cost = std::get<1>(*wait_it);
          const bool tied_before_random =
              std::get<0>(top_cost) == std::get<0>(wait_cost) &&
              std::fabs(std::get<1>(top_cost) - std::get<1>(wait_cost)) <=
                  1e-6f;
          if (tied_before_random) {
            ++goal_departures_top_leave_tied_wait;
          } else {
            ++goal_departures_top_leave_strict;
          }
        }
      }
    }
  }

  // cleanup
  for (auto i = 0; i < N; ++i) {
    occupied_now[Q_from[i]->id] = NO_AGENT;
    if (Q_to[i] != nullptr) occupied_next[Q_to[i]->id] = NO_AGENT;
  }

  return success;
}

bool PIBT::funcPIBT(const int i, const Config &Q_from, Config &Q_to,
                    const std::set<int> &default_policy_agents,
                    const std::vector<std::vector<int>>
                        *forbidden_vertex_ids)
{
  // Track recursion metrics
  recursion_count++;
  current_depth++;
  if (current_depth > max_recursion_depth) {
    max_recursion_depth = current_depth;
  }
  if (goal_diag_enabled &&
      occupied_next[Q_from[i]->id] != NO_AGENT &&
      occupied_next[Q_from[i]->id] != i) {
    displaced_on_entry[i] = true;
  }

  const auto K = Q_from[i]->neighbor.size();

  // emulate swap
  auto swap_agent = NO_AGENT;
  if (!policy.use_model ||
      default_policy_agents.find(i) != default_policy_agents.end()) {
    swap_agent =
        is_swap_required_and_possible(i, Q_from, Q_to, policy.get(i, 0));
    if (swap_agent != NO_AGENT) {
      std::reverse(policy.preferences[i].begin(),
                   policy.preferences[i].begin() + K + 1);
    }
  }
  auto swap_operation = [&]() {
    if (swap_agent != NO_AGENT &&                 // swap_agent exists
        Q_to[swap_agent] == nullptr &&            // not decided
        occupied_next[Q_from[i]->id] == NO_AGENT  // free
    ) {
      // pull swap_agent
      occupied_next[Q_from[i]->id] = swap_agent;
      Q_to[swap_agent] = Q_from[i];
    }
  };

  // main loop
  for (size_t k = 0; k < K + 1; ++k) {
    const auto u = policy.get(i, k);
    if (forbidden_vertex_ids != nullptr) {
      const auto &forbidden = (*forbidden_vertex_ids)[i];
      if (std::find(forbidden.begin(), forbidden.end(), u->id) !=
          forbidden.end()) {
        continue;
      }
    }

    // avoid vertex conflicts
    if (occupied_next[u->id] != NO_AGENT) continue;

    const auto j = occupied_now[u->id];

    // avoid swap conflicts with constraints
    if (j != NO_AGENT && Q_to[j] == Q_from[i]) continue;

    // reserve next location
    occupied_next[u->id] = i;
    Q_to[i] = u;

    // priority inheritance
    if (j != NO_AGENT && u != Q_from[i] && Q_to[j] == nullptr &&
        !funcPIBT(j, Q_from, Q_to, default_policy_agents,
                  forbidden_vertex_ids)) {
      continue;
    }

    // success to plan next one step
    if (goal_diag_enabled) chosen_rank[i] = static_cast<int>(k);
    if (k == 0) swap_operation();
    current_depth--;  // Decrement depth before returning
    return true;
  }

  // failed to secure node
  occupied_next[Q_from[i]->id] = i;
  Q_to[i] = Q_from[i];
  current_depth--;  // Decrement depth before returning
  return false;
}

int PIBT::is_swap_required_and_possible(const int i, const Config &Q_from,
                                        Config &Q_to, Vertex *v_i_target)
{
  if (!SWAP) return NO_AGENT;
  // agent-j occupying the desired vertex for agent-i
  const auto j = occupied_now[v_i_target->id];
  if (j != NO_AGENT && j != i &&  // j exists
      Q_to[j] == nullptr &&       // j does not decide next location
      is_swap_required(i, j, Q_from[i], Q_from[j]) &&  // swap required
      is_swap_possible(Q_from[j], Q_from[i])           // swap possible
  ) {
    return j;
  }

  // for clear operation, c.f., push & swap
  if (v_i_target != Q_from[i]) {
    for (auto u : Q_from[i]->neighbor) {
      const auto k = occupied_now[u->id];
      if (k != NO_AGENT &&            // k exists
          v_i_target != Q_from[k] &&  // this is for clear operation
          is_swap_required(k, i, Q_from[i],
                           v_i_target) &&  // emulating from one step ahead
          is_swap_possible(v_i_target, Q_from[i])) {
        return k;
      }
    }
  }
  return NO_AGENT;
}

bool PIBT::is_swap_required(const int pusher, const int puller,
                            Vertex *v_pusher_origin, Vertex *v_puller_origin)
{
  auto v_pusher = v_pusher_origin;
  auto v_puller = v_puller_origin;
  Vertex *tmp = nullptr;
  while (D->get(pusher, v_puller) < D->get(pusher, v_pusher)) {
    auto n = v_puller->neighbor.size();
    // remove agents who need not to move
    for (auto u : v_puller->neighbor) {
      const auto i = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && i != NO_AGENT && ins->goals[i] == u)) {
        --n;
      } else {
        tmp = u;
      }
    }
    if (n >= 2) return false;  // able to swap at v_l
    if (n <= 0) break;
    v_pusher = v_puller;
    v_puller = tmp;
  }

  return (D->get(puller, v_pusher) < D->get(puller, v_puller)) &&
         (D->get(pusher, v_pusher) == 0 ||
          D->get(pusher, v_puller) < D->get(pusher, v_pusher));
}

bool PIBT::is_swap_possible(Vertex *v_pusher_origin, Vertex *v_puller_origin)
{
  // simulate pull
  auto v_pusher = v_pusher_origin;
  auto v_puller = v_puller_origin;
  Vertex *tmp = nullptr;
  while (v_puller != v_pusher_origin) {  // avoid loop
    auto n = v_puller->neighbor.size();
    for (auto u : v_puller->neighbor) {
      const auto i = occupied_now[u->id];
      if (u == v_pusher ||
          (u->neighbor.size() == 1 && i != NO_AGENT && ins->goals[i] == u)) {
        --n;
      } else {
        tmp = u;
      }
    }
    if (n >= 2) return true;  // able to swap at v_next
    if (n <= 0) return false;
    v_pusher = v_puller;
    v_puller = tmp;
  }
  return false;
}

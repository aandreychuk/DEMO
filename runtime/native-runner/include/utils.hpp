/*
 * utility functions
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <climits>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <regex>
#include <set>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils.hpp"

using Time = std::chrono::steady_clock;
using uint = unsigned int;

// time manager
struct Deadline {
  const Time::time_point t_s;
  const double time_limit_ms;

  Deadline(double _time_limit_ms = 0);
  double elapsed_ms() const;
  double elapsed_ns() const;
};

double elapsed_ms(const Deadline *deadline);
double elapsed_ns(const Deadline *deadline);
double elapsed_ms(const Deadline &deadline);
bool is_expired(const Deadline *deadline);
bool is_expired(const Deadline &deadline);

float get_random_float(std::mt19937 &MT, float from = 0, float to = 1);
float get_random_float(std::mt19937 *MT, float from = 0, float to = 1);
int get_random_int(std::mt19937 &MT, int from = 0, int to = 1);
int get_random_int(std::mt19937 *MT, int from = 0, int to = 1);

template <typename Head, typename... Tail>
void _info(const int level, const int verbose, Head &&head, Tail &&...tail);

void _info(const int level, const int verbose);

template <typename Head, typename... Tail>
void _info(const int level, const int verbose, Head &&head, Tail &&...tail)
{
  if (verbose < level) return;
  std::cout << head;
  _info(level, verbose, std::forward<Tail>(tail)...);
}

std::ostream &operator<<(std::ostream &os, const std::vector<int> &arr);
std::ostream &operator<<(std::ostream &os, const std::list<int> &arr);
std::ostream &operator<<(std::ostream &os, const std::set<int> &arr);

namespace Common
{
  extern Deadline *DEADLINE;
  extern int VERBOSE;
  extern int MODEL_LOAD_MS;

  // Runtime & policy / search statistics for LaGAT / LaCAM
  struct LaGATStats {
    std::atomic<long long> highlevel_iterations{0};       // LaCAM outer-loop iterations
    std::atomic<long long> generated_nodes{0};            // number of HNodes created
    std::atomic<long long> lowlevel_nodes_popped{0};      // LNodes expanded
    std::atomic<long long> lowlevel_constraints_generated{0};
    std::atomic<long long> successor_attempts{0};         // PIBT successor requests
    std::atomic<long long> successor_failures{0};         // no valid successor
    std::atomic<long long> duplicate_successors{0};       // successor already explored
    std::atomic<long long> open_node_exhaustions{0};      // HNode has no LNodes left
    std::atomic<long long> bound_restarts{0};
    std::atomic<long long> random_restarts{0};
    std::atomic<long long> goal_restarts{0};

    // Deadlock-detection fallback to heuristic PIBT
    std::atomic<long long> dd_ancestor_checks{0};
    std::atomic<long long> dd_trigger_events{0};
    std::atomic<long long> dd_fallback_agents_added{0};

    // Policy usage
    std::atomic<long long> policy_setpref_calls{0};       // AgentPolicy::set_preferences calls
    std::atomic<long long> policy_forward_calls{0};       // TorchScript forward() calls
    std::atomic<long long> policy_forward_ms{0};          // time spent in forward() [ms]

    // Action attribution: learned policy vs default/PIBT
    std::atomic<long long> actions_policy_all{0};         // across all generated nodes
    std::atomic<long long> actions_naive_all{0};
    std::atomic<long long> actions_policy_solution{0};    // restricted to final solution path
    std::atomic<long long> actions_naive_solution{0};

    LaGATStats() = default;
    LaGATStats(const LaGATStats&) = delete;
    LaGATStats& operator=(const LaGATStats&) = delete;
  };

  extern LaGATStats STATS;

  void set_print_options(Deadline &_deadline, int _verbose = 0);

  // global
  template <typename... Body>
  void info(const int level, Body &&...body)
  {
    if (VERBOSE < level) return;
    _info(level, VERBOSE, (body)...);
  }
};  // namespace Common

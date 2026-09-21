#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <fstream>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <torch/torch.h>

#include "dist_table.hpp"
#include "instance.hpp"
#include "pibt.hpp"
#include "policy.hpp"
#include "utils.hpp"

namespace {

struct Arguments {
  std::string map;
  std::string scen;
  std::string model;
  std::string policy = "dmm";
  std::string mode = "soft";
  int num_agents = 1;
  int max_steps = 1;
  int seed = 0;
  std::string sampling = "deterministic";
  float tau = 1.0f;
  bool escape_repeated_states = false;
  int max_repeat_retries = 16;
  int repeat_history_window = 0;
  bool diagnostics = false;
  int trace_window = 256;
  std::string output_prefix;
  std::string dump_training;
  int dump_training_stride = 1;
  bool dump_training_repeat_only = false;
  bool dump_training_fallback_only = false;
  std::string fallback_model;
  int fallback_after_stagnation = 0;
  int fallback_max_remaining = 0;
  bool fallback_release_on_progress = false;
  int fallback_min_burst = 1;
  int fallback_budget_reserve = 0;
  int fallback_urgent_remaining = 0;
};

Arguments parse_arguments(int argc, char** argv)
{
  Arguments args;
  auto value = [&](int& i) -> std::string {
    if (++i >= argc) throw std::runtime_error("missing argument value");
    return argv[i];
  };
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--map") args.map = value(i);
    else if (key == "--scen") args.scen = value(i);
    else if (key == "--model") args.model = value(i);
    else if (key == "--policy") args.policy = value(i);
    else if (key == "--mode") args.mode = value(i);
    else if (key == "--num-agents") args.num_agents = std::stoi(value(i));
    else if (key == "--max-steps") args.max_steps = std::stoi(value(i));
    else if (key == "--seed") args.seed = std::stoi(value(i));
    else if (key == "--sampling") args.sampling = value(i);
    else if (key == "--tau") args.tau = std::stof(value(i));
    else if (key == "--escape-repeated-states")
      args.escape_repeated_states = true;
    else if (key == "--max-repeat-retries")
      args.max_repeat_retries = std::stoi(value(i));
    else if (key == "--repeat-history-window")
      args.repeat_history_window = std::stoi(value(i));
    else if (key == "--diagnostics") args.diagnostics = true;
    else if (key == "--trace-window")
      args.trace_window = std::stoi(value(i));
    else if (key == "--output-prefix" || key == "--trajectory-prefix")
      args.output_prefix = value(i);
    else if (key == "--dump-training")
      args.dump_training = value(i);
    else if (key == "--dump-training-stride")
      args.dump_training_stride = std::stoi(value(i));
    else if (key == "--dump-training-repeat-only")
      args.dump_training_repeat_only = true;
    else if (key == "--dump-training-fallback-only")
      args.dump_training_fallback_only = true;
    else if (key == "--fallback-model")
      args.fallback_model = value(i);
    else if (key == "--fallback-after-stagnation")
      args.fallback_after_stagnation = std::stoi(value(i));
    else if (key == "--fallback-max-remaining")
      args.fallback_max_remaining = std::stoi(value(i));
    else if (key == "--fallback-release-on-progress")
      args.fallback_release_on_progress = true;
    else if (key == "--fallback-min-burst")
      args.fallback_min_burst = std::stoi(value(i));
    else if (key == "--fallback-budget-reserve")
      args.fallback_budget_reserve = std::stoi(value(i));
    else if (key == "--fallback-urgent-remaining")
      args.fallback_urgent_remaining = std::stoi(value(i));
    else throw std::runtime_error("unknown argument: " + key);
  }
  if (args.map.empty() || args.scen.empty() || args.model.empty() ||
      args.num_agents <= 0 || args.max_steps <= 0 ||
      args.trace_window <= 0 ||
      args.dump_training_stride <= 0 ||
      args.max_repeat_retries <= 0 ||
      args.repeat_history_window < 0 ||
      args.fallback_after_stagnation < 0 ||
      args.fallback_max_remaining < 0 ||
      args.fallback_min_burst <= 0 ||
      args.fallback_budget_reserve < 0 ||
      args.fallback_urgent_remaining < 0 ||
      (args.fallback_after_stagnation > 0 &&
       args.fallback_model.empty()) ||
      args.tau <= 0 ||
      (args.policy != "dmm" && args.policy != "magat" &&
       args.policy != "lc_mapf") ||
      (args.sampling != "deterministic" &&
       args.sampling != "probabilistic") ||
      (args.mode != "soft" && args.mode != "pibt") ||
      (args.mode == "soft" && args.policy != "dmm")) {
    throw std::runtime_error(
        "usage: dmm_rollout --map MAP --scen SCEN --model MODEL "
        "--policy dmm|magat|lc_mapf --num-agents N --max-steps T "
        "--mode soft|pibt [--seed S] "
        "[--sampling deterministic|probabilistic] [--tau T] "
        "[--escape-repeated-states] [--max-repeat-retries N] "
        "[--repeat-history-window N (0 means all states)] "
        "[--dump-training FILE] [--dump-training-stride N] "
        "[--dump-training-repeat-only] "
        "[--dump-training-fallback-only] "
        "[--fallback-model MODEL --fallback-after-stagnation N "
        "--fallback-max-remaining N] [--fallback-release-on-progress] "
        "[--fallback-min-burst N] [--fallback-budget-reserve N] "
        "[--fallback-urgent-remaining N]");
  }
  return args;
}

int action_index(const Vertex* from, const Vertex* to)
{
  const int dx = to->x - from->x;
  const int dy = to->y - from->y;
  if (dx == 0 && dy == 0) return 0;
  if (dx == 0 && dy == -1) return 1;
  if (dx == 0 && dy == 1) return 2;
  if (dx == -1 && dy == 0) return 3;
  if (dx == 1 && dy == 0) return 4;
  throw std::runtime_error("non-adjacent DMM transition");
}

long long cell_key(int x, int y)
{
  const auto packed =
      (static_cast<unsigned long long>(static_cast<unsigned int>(y)) << 32) |
      static_cast<unsigned int>(x);
  return static_cast<long long>(packed);
}

struct Edge {
  int x0;
  int y0;
  int x1;
  int y1;
  bool operator==(const Edge& other) const
  {
    return x0 == other.x0 && y0 == other.y0 &&
           x1 == other.x1 && y1 == other.y1;
  }
};

struct EdgeHash {
  size_t operator()(const Edge& edge) const
  {
    size_t h = 1469598103934665603ULL;
    for (const int value : {edge.x0, edge.y0, edge.x1, edge.y1}) {
      h ^= static_cast<unsigned int>(value);
      h *= 1099511628211ULL;
    }
    return h;
  }
};

Config apply_soft_collisions(const Instance& ins, const Config& current,
                             std::vector<int>& actions,
                             long long& agent_collisions,
                             long long& obstacle_collisions)
{
  static const std::array<int, 5> dx = {0, 0, 0, -1, 1};
  static const std::array<int, 5> dy = {0, -1, 1, 0, 0};
  std::unordered_map<long long, std::vector<int>> used_cells;
  std::unordered_map<Edge, std::vector<int>, EdgeHash> used_edges;

  for (int i = 0; i < static_cast<int>(current.size()); ++i) {
    const int x = current[i]->x;
    const int y = current[i]->y;
    const int nx = x + dx[actions[i]];
    const int ny = y + dy[actions[i]];
    used_cells[cell_key(nx, ny)].push_back(i);
    used_edges[{x, y, nx, ny}].push_back(i);
    if (nx != x || ny != y) {
      used_edges[{nx, ny, x, y}].push_back(i);
    }
  }

  for (int i = 0; i < static_cast<int>(current.size()); ++i) {
    const int x = current[i]->x;
    const int y = current[i]->y;
    const int nx = x + dx[actions[i]];
    const int ny = y + dy[actions[i]];
    if (used_edges[{x, y, nx, ny}].size() > 1) {
      auto& desired = used_cells[cell_key(nx, ny)];
      desired.erase(std::find(desired.begin(), desired.end(), i));
      used_cells[cell_key(x, y)].push_back(i);
      actions[i] = 0;
      ++agent_collisions;
    }
  }

  std::function<void(int, long long)> revert =
      [&](int agent, long long desired_key) {
        actions[agent] = 0;
        auto& desired = used_cells[desired_key];
        const auto it = std::find(desired.begin(), desired.end(), agent);
        if (it != desired.end()) desired.erase(it);
        const long long origin =
            cell_key(current[agent]->x, current[agent]->y);
        auto& origin_users = used_cells[origin];
        if (!origin_users.empty()) {
          const int displaced = origin_users.front();
          origin_users.push_back(agent);
          revert(displaced, origin);
        } else {
          origin_users.push_back(agent);
        }
      };

  for (int i = static_cast<int>(current.size()) - 1; i >= 0; --i) {
    const int x = current[i]->x;
    const int y = current[i]->y;
    const int nx = x + dx[actions[i]];
    const int ny = y + dy[actions[i]];
    const bool obstacle =
        nx < 0 || nx >= ins.G->width || ny < 0 || ny >= ins.G->height ||
        ins.G->U[ins.G->width * ny + nx] == nullptr;
    if (used_cells[cell_key(nx, ny)].size() > 1 || obstacle) {
      if (obstacle) ++obstacle_collisions;
      else ++agent_collisions;
      revert(i, cell_key(nx, ny));
    }
  }

  Config next(current.size(), nullptr);
  for (int i = 0; i < static_cast<int>(current.size()); ++i) {
    const int nx = current[i]->x + dx[actions[i]];
    const int ny = current[i]->y + dy[actions[i]];
    next[i] = ins.G->U[ins.G->width * ny + nx];
  }
  return next;
}

bool is_goal(const Instance& ins, const Config& config)
{
  return std::equal(config.begin(), config.end(), ins.goals.begin());
}

using ConfigKey = std::vector<int>;

struct ConfigKeyHash {
  size_t operator()(const ConfigKey& key) const
  {
    size_t hash = 1469598103934665603ULL;
    for (const int vertex_id : key) {
      hash ^= static_cast<unsigned int>(vertex_id);
      hash *= 1099511628211ULL;
    }
    return hash;
  }
};

ConfigKey make_config_key(const Config& config)
{
  ConfigKey key;
  key.reserve(config.size());
  for (const Vertex* vertex : config) key.push_back(vertex->id);
  return key;
}

}  // namespace

int main(int argc, char** argv)
{
  try {
    const Arguments args = parse_arguments(argc, argv);
    if (args.diagnostics) {
      setenv("LAGAT_PIBT_GOAL_DIAG", "1", 1);
    }
    torch::manual_seed(args.seed);
    if (torch::cuda::is_available()) torch::cuda::manual_seed(args.seed);
    at::globalContext().setBenchmarkCuDNN(false);
    at::globalContext().setDeterministicAlgorithms(true, false);

    const Instance ins(args.scen, args.map, args.num_agents);
    if (!ins.is_valid()) throw std::runtime_error("invalid MAPF instance");
    DistTable distances(&ins);
    PolicyConfig config;
    config.policy_type = args.policy;
    if (args.policy == "dmm") {
      config.dmm_model_path = args.model;
      config.dmm_device = "cuda";
    } else if (args.policy == "magat") {
      config.model_filepath = args.model;
    } else {
      config.lc_mapf_model_path = args.model;
      config.lc_mapf_device = "cuda";
    }
    AgentPolicy::SAMPLING_TEMPERTURE = args.tau;
    // The established native DMM POGEMA setup samples communication votes,
    // while this switch controls only the final PIBT action ranking.
    AgentPolicy::SAMPLING_STRATEGY =
        args.sampling == "probabilistic"
            ? AgentPolicy::Probablistic
            : AgentPolicy::Deterministic;

    std::unique_ptr<AgentPolicy> raw_policy;
    std::unique_ptr<PIBT> pibt;
    std::unique_ptr<PIBT> fallback_pibt;
    if (args.mode == "soft") raw_policy =
        std::make_unique<AgentPolicy>(&ins, &distances, config, args.seed);
    else {
      pibt = std::make_unique<PIBT>(&ins, &distances, config, args.seed);
      if (!args.fallback_model.empty()) {
        PolicyConfig fallback_config = config;
        fallback_config.dmm_model_path = args.fallback_model;
        fallback_pibt = std::make_unique<PIBT>(
            &ins, &distances, fallback_config, args.seed);
      }
    }

    Config current = ins.starts;
    std::ofstream trajectory;
    std::ofstream decisions;
    std::ofstream training_dump;
    if (!args.output_prefix.empty()) {
      trajectory.open(args.output_prefix + ".trajectory.tsv");
      decisions.open(args.output_prefix + ".decisions.tsv");
      if (!trajectory || !decisions) {
        throw std::runtime_error("failed to open trajectory trace files");
      }
      trajectory << "step\tagent\tx\ty\n";
      decisions << "step\tagent\tpre_x\tpre_y\tpreferred0\tpreferred1"
                   "\tpreferred2\tpreferred3\tpreferred4\texecuted"
                   "\tchosen_rank\tpriority\torder_rank\n";
      for (int i = 0; i < static_cast<int>(ins.N); ++i) {
        trajectory << 0 << '\t' << i << '\t' << current[i]->x << '\t'
                   << current[i]->y << '\n';
      }
    }
    if (!args.dump_training.empty()) {
      training_dump.open(args.dump_training, std::ios::binary);
      if (!training_dump) {
        throw std::runtime_error("failed to open training dump");
      }
      const std::array<char, 8> magic = {'D', 'M', 'M', 'T', 'R', 'N', '1', '\0'};
      const uint32_t version = 1;
      const uint32_t agents = static_cast<uint32_t>(ins.N);
      const uint32_t observation_tokens = 256;
      const uint32_t chat_slots = 13;
      training_dump.write(magic.data(), magic.size());
      training_dump.write(reinterpret_cast<const char*>(&version), sizeof(version));
      training_dump.write(reinterpret_cast<const char*>(&agents), sizeof(agents));
      training_dump.write(
          reinterpret_cast<const char*>(&observation_tokens),
          sizeof(observation_tokens));
      training_dump.write(
          reinterpret_cast<const char*>(&chat_slots), sizeof(chat_slots));
    }
    std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>> history(
        ins.N);
    for (auto& row : history) row.fill(-1);
    std::vector<float> priorities(ins.N);
    std::vector<int> order(ins.N);
    for (int i = 0; i < static_cast<int>(ins.N); ++i) {
      priorities[i] = static_cast<float>(distances.get(i, current[i])) / 10000;
      order[i] = i;
    }
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return priorities[a] > priorities[b]; });

    std::vector<int> solve_time(ins.N, -1);
    std::vector<int> last_move_step(ins.N, -1);
    std::deque<ConfigKey> recent_configs;
    std::unordered_map<ConfigKey, int, ConfigKeyHash> recent_config_counts;
    std::unordered_set<ConfigKey, ConfigKeyHash> visited;
    auto remember_config = [&](const Config& config) {
      ConfigKey key = make_config_key(config);
      if (args.repeat_history_window == 0) {
        visited.insert(std::move(key));
        return;
      }
      recent_configs.push_back(key);
      ++recent_config_counts[key];
      while (static_cast<int>(recent_configs.size()) >
             args.repeat_history_window) {
        ConfigKey expired = std::move(recent_configs.front());
        recent_configs.pop_front();
        auto it = recent_config_counts.find(expired);
        if (--it->second == 0) recent_config_counts.erase(it);
      }
    };
    auto was_visited = [&](const Config& config) {
      ConfigKey key = make_config_key(config);
      if (args.repeat_history_window == 0) {
        return visited.find(key) != visited.end();
      }
      return recent_config_counts.find(key) != recent_config_counts.end();
    };
    remember_config(current);
    long long repeated_candidates = 0;
    long long repeat_retry_attempts = 0;
    long long repeat_constraints = 0;
    long long repeat_escapes = 0;
    long long repeat_unescaped = 0;
    long long repeat_detection_steps = 0;
    long long argmax_conflict_steps = 0;
    long long argmax_vertex_conflict_groups = 0;
    long long argmax_swap_conflicts = 0;
    long long argmax_conflicting_actions = 0;
    long long pibt_shield_steps = 0;
    long long pibt_shielded_actions = 0;
    std::vector<Vertex*> raw_top_targets(ins.N, nullptr);
    std::vector<int> raw_target_counts(ins.G->size(), 0);
    std::vector<int> current_occupant(ins.G->size(), -1);
    std::vector<char> raw_conflicting(ins.N, false);
    const size_t trace_size =
        args.diagnostics
            ? static_cast<size_t>(args.trace_window) * ins.N
            : 0;
    std::vector<int> trace_pre(trace_size, -1);
    std::vector<int> trace_post(trace_size, -1);
    std::vector<signed char> trace_preferred(trace_size, -1);
    std::vector<signed char> trace_executed(trace_size, -1);
    std::vector<short> trace_rank(trace_size, -1);
    long long agent_collisions = 0;
    long long obstacle_collisions = 0;
    int episode_steps = 0;
    const auto started = std::chrono::steady_clock::now();
    bool solved = is_goal(ins, current);
    int best_reached = 0;
    for (int i = 0; i < static_cast<int>(ins.N); ++i) {
      if (current[i] == ins.goals[i]) ++best_reached;
    }
    int last_progress_step = 0;
    bool fallback_active = false;
    bool fallback_ever_active = false;
    int fallback_activation_step = -1;
    long long fallback_activations = 0;
    long long fallback_steps = 0;
    int fallback_burst_steps = 0;

    for (int step = 0; step < args.max_steps && !solved; ++step) {
      std::vector<int> actions(ins.N, 0);
      Config next(ins.N, nullptr);
      if (args.mode == "soft") {
        const auto probs =
            raw_policy->get_dmm_action_probabilities(current, &history);
        for (int i = 0; i < static_cast<int>(ins.N); ++i) {
          actions[i] = static_cast<int>(
              std::max_element(probs[i].begin(), probs[i].end()) -
              probs[i].begin());
        }
        for (int i = 0; i < static_cast<int>(ins.N); ++i) {
          std::rotate(history[i].begin(), history[i].begin() + 1,
                      history[i].end());
          history[i].back() = actions[i];
        }
        next = apply_soft_collisions(
            ins, current, actions, agent_collisions, obstacle_collisions);
      } else {
        const int remaining_at_step_start =
            static_cast<int>(ins.N) - best_reached;
        if (!fallback_active && fallback_pibt &&
            step - last_progress_step >= args.fallback_after_stagnation &&
            (args.fallback_budget_reserve == 0 ||
             args.max_steps - step <= args.fallback_budget_reserve ||
             (args.fallback_urgent_remaining > 0 &&
              remaining_at_step_start <= args.fallback_urgent_remaining)) &&
            (args.fallback_max_remaining == 0 ||
             remaining_at_step_start <= args.fallback_max_remaining)) {
          fallback_active = true;
          fallback_ever_active = true;
          fallback_activation_step = step;
          ++fallback_activations;
          fallback_burst_steps = 0;
        }
        PIBT* active_pibt =
            fallback_active ? fallback_pibt.get() : pibt.get();
        if (fallback_active) {
          ++fallback_steps;
          ++fallback_burst_steps;
        }
        std::vector<std::vector<int>> forbidden_vertex_ids(ins.N);
        Config repeated_escape_candidate;
        std::vector<std::vector<int>> dump_observations;
        std::vector<std::vector<int>> dump_chat;
        std::vector<uint8_t> dump_preferred;
        bool dump_pending = false;
        bool escape_candidate_moves = false;
        int escape_agent = -1;
        for (const int agent : order) {
          if (current[agent] != ins.goals[agent]) {
            escape_agent = agent;
            break;
          }
        }
        bool saw_repeated_candidate = false;
        int retries = 0;
        while (true) {
          next.assign(ins.N, nullptr);
          const auto* forbidden =
              retries == 0 ? nullptr : &forbidden_vertex_ids;
          if (!active_pibt->set_new_config(current, next, order, {}, &history,
                                           forbidden)) {
            if (repeated_escape_candidate.empty()) {
              throw std::runtime_error(
                  "PIBT failed to produce a joint action");
            }
            next = std::move(repeated_escape_candidate);
            ++repeat_unescaped;
            break;
          }

          if (retries == 0) {
            if (training_dump && step % args.dump_training_stride == 0) {
              active_pibt->policy.get_dmm_encoded_inputs(dump_observations,
                                                          dump_chat);
              dump_preferred.resize(ins.N);
              for (int i = 0; i < static_cast<int>(ins.N); ++i) {
                dump_preferred[i] = static_cast<uint8_t>(
                    action_index(current[i], active_pibt->policy.get(i, 0)));
              }
              dump_pending = true;
            }
            std::fill(raw_conflicting.begin(), raw_conflicting.end(), false);
            std::vector<int> touched_targets;
            touched_targets.reserve(ins.N);
            for (int i = 0; i < static_cast<int>(ins.N); ++i) {
              current_occupant[current[i]->id] = i;
              raw_top_targets[i] = active_pibt->policy.get(i, 0);
              const int target_id = raw_top_targets[i]->id;
              if (raw_target_counts[target_id]++ == 0) {
                touched_targets.push_back(target_id);
              }
            }
            long long vertex_groups_this_step = 0;
            for (const int target_id : touched_targets) {
              if (raw_target_counts[target_id] > 1) {
                ++vertex_groups_this_step;
              }
            }
            long long swap_conflicts_this_step = 0;
            for (int i = 0; i < static_cast<int>(ins.N); ++i) {
              if (raw_target_counts[raw_top_targets[i]->id] > 1) {
                raw_conflicting[i] = true;
              }
              const int j = current_occupant[raw_top_targets[i]->id];
              if (j > i && raw_top_targets[i] == current[j] &&
                  raw_top_targets[j] == current[i] &&
                  raw_top_targets[i] != current[i]) {
                ++swap_conflicts_this_step;
                raw_conflicting[i] = true;
                raw_conflicting[j] = true;
              }
            }
            const long long conflicting_actions_this_step =
                std::count(raw_conflicting.begin(), raw_conflicting.end(),
                           true);
            if (vertex_groups_this_step > 0 ||
                swap_conflicts_this_step > 0) {
              ++argmax_conflict_steps;
            }
            argmax_vertex_conflict_groups += vertex_groups_this_step;
            argmax_swap_conflicts += swap_conflicts_this_step;
            argmax_conflicting_actions += conflicting_actions_this_step;

            long long shielded_actions_this_step = 0;
            for (int i = 0; i < static_cast<int>(ins.N); ++i) {
              if (next[i] != raw_top_targets[i]) {
                ++shielded_actions_this_step;
              }
              current_occupant[current[i]->id] = -1;
            }
            for (const int target_id : touched_targets) {
              raw_target_counts[target_id] = 0;
            }
            if (shielded_actions_this_step > 0) ++pibt_shield_steps;
            pibt_shielded_actions += shielded_actions_this_step;
          }

          const bool repeated =
              args.escape_repeated_states &&
              was_visited(next);
          if (!repeated) {
            if (saw_repeated_candidate) ++repeat_escapes;
            break;
          }

          if (!saw_repeated_candidate) ++repeat_detection_steps;
          saw_repeated_candidate = true;
          ++repeated_candidates;
          const bool candidate_moves =
              escape_agent >= 0 &&
              next[escape_agent] != current[escape_agent];
          if (repeated_escape_candidate.empty() ||
              (!escape_candidate_moves && candidate_moves)) {
            repeated_escape_candidate = next;
            escape_candidate_moves = candidate_moves;
          }
          if (retries >= args.max_repeat_retries) {
            next = std::move(repeated_escape_candidate);
            ++repeat_unescaped;
            break;
          }

          bool added_constraint = false;
          for (const int agent : order) {
            if (current[agent] == ins.goals[agent]) continue;
            auto& forbidden_for_agent = forbidden_vertex_ids[agent];
            const int target_id = next[agent]->id;
            if (std::find(forbidden_for_agent.begin(),
                          forbidden_for_agent.end(),
                          target_id) != forbidden_for_agent.end()) {
              continue;
            }
            const size_t action_count =
                current[agent]->neighbor.size() + 1;
            if (forbidden_for_agent.size() + 1 >= action_count) continue;
            forbidden_for_agent.push_back(target_id);
            ++repeat_constraints;
            added_constraint = true;
            break;
          }
          if (!added_constraint) {
            next = std::move(repeated_escape_candidate);
            ++repeat_unescaped;
            break;
          }
          ++retries;
          ++repeat_retry_attempts;
        }
        if (dump_pending &&
            (!args.dump_training_repeat_only || saw_repeated_candidate) &&
            (!args.dump_training_fallback_only || fallback_active)) {
          const uint32_t dump_step = static_cast<uint32_t>(step);
          training_dump.write(reinterpret_cast<const char*>(&dump_step),
                              sizeof(dump_step));
          for (int i = 0; i < static_cast<int>(ins.N); ++i) {
            for (const int token : dump_observations[i]) {
              const uint8_t value = static_cast<uint8_t>(token);
              training_dump.write(reinterpret_cast<const char*>(&value),
                                  sizeof(value));
            }
          }
          for (int i = 0; i < static_cast<int>(ins.N); ++i) {
            for (const int neighbor : dump_chat[i]) {
              const int16_t value = static_cast<int16_t>(neighbor);
              training_dump.write(reinterpret_cast<const char*>(&value),
                                  sizeof(value));
            }
          }
          training_dump.write(
              reinterpret_cast<const char*>(dump_preferred.data()),
              dump_preferred.size() * sizeof(dump_preferred.front()));
        }
        std::vector<int> order_rank;
        if (decisions) {
          order_rank.resize(ins.N);
          for (int rank = 0; rank < static_cast<int>(ins.N); ++rank) {
            order_rank[order[rank]] = rank;
          }
        }
        for (int i = 0; i < static_cast<int>(ins.N); ++i) {
          actions[i] = action_index(current[i], next[i]);
          if (decisions) {
            decisions << step << '\t' << i << '\t' << current[i]->x << '\t'
                      << current[i]->y;
            const size_t valid_actions = current[i]->neighbor.size() + 1;
            for (size_t rank = 0; rank < 5; ++rank) {
              decisions << '\t';
              if (rank < valid_actions) {
                decisions << action_index(
                    current[i], active_pibt->policy.get(i, rank));
              } else {
                decisions << -1;
              }
            }
            decisions << '\t' << actions[i]
                      << '\t' << active_pibt->chosen_rank[i]
                      << '\t' << priorities[i]
                      << '\t' << order_rank[i] << '\n';
          }
          if (args.diagnostics) {
            const size_t offset =
                static_cast<size_t>(step % args.trace_window) * ins.N + i;
            trace_pre[offset] = current[i]->id;
            trace_post[offset] = next[i]->id;
            trace_preferred[offset] =
                static_cast<signed char>(
                    action_index(current[i], active_pibt->policy.get(i, 0)));
            trace_executed[offset] =
                static_cast<signed char>(actions[i]);
            trace_rank[offset] =
                static_cast<short>(active_pibt->chosen_rank[i]);
          }
          std::rotate(history[i].begin(), history[i].begin() + 1,
                      history[i].end());
          history[i].back() = actions[i];
        }
      }

      current = std::move(next);
      remember_config(current);
      episode_steps = step + 1;
      if (trajectory) {
        for (int i = 0; i < static_cast<int>(ins.N); ++i) {
          trajectory << episode_steps << '\t' << i << '\t'
                     << current[i]->x << '\t' << current[i]->y << '\n';
        }
      }
      for (int i = 0; i < static_cast<int>(ins.N); ++i) {
        if (actions[i] != 0) last_move_step[i] = episode_steps;
        if (current[i] == ins.goals[i]) {
          if (solve_time[i] < 0) solve_time[i] = step;
        } else {
          solve_time[i] = -1;
        }
      }
      int reached_this_step = 0;
      for (int i = 0; i < static_cast<int>(ins.N); ++i) {
        if (current[i] == ins.goals[i]) ++reached_this_step;
      }
      if (reached_this_step > best_reached) {
        best_reached = reached_this_step;
        last_progress_step = episode_steps;
        if (fallback_active && args.fallback_release_on_progress &&
            fallback_burst_steps >= args.fallback_min_burst) {
          fallback_active = false;
        }
      }
      if (args.mode == "pibt") {
        for (int i = 0; i < static_cast<int>(ins.N); ++i) {
          if (current[i] != ins.goals[i]) priorities[i] += 1;
          else priorities[i] -= std::floor(priorities[i]);
        }
        // LaGAT creates a fresh LoopNodeState for every newly generated
        // configuration.  Its priority order therefore always starts from
        // agent-index order before sorting; sorting the previous timestep's
        // order changes the implicit tie-breaking for equal priorities.
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int a, int b) {
                    return priorities[a] > priorities[b];
                  });
      }
      solved = is_goal(ins, current);
    }

    const double runtime_sec = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    int reached = 0;
    long long soc = 0;
    int makespan = 0;
    for (int i = 0; i < static_cast<int>(ins.N); ++i) {
      if (current[i] == ins.goals[i]) ++reached;
      if (solve_time[i] >= 0) {
        soc += solve_time[i] + 1;
        makespan = std::max(makespan, solve_time[i] + 1);
      }
    }
    std::cout << "status=" << (solved ? "solved" : "no_solution") << '\n'
              << "solved=" << (solved ? 1 : 0) << '\n'
              << "csr=" << (solved ? 1 : 0) << '\n'
              << "isr=" << static_cast<double>(reached) / ins.N << '\n'
              << "soc=" << (solved ? soc : -1) << '\n'
              << "makespan=" << (solved ? makespan : -1) << '\n'
              << "episode_steps=" << episode_steps << '\n'
              << "runtime_sec=" << runtime_sec << '\n'
              << "policy_forward_ms=" << Common::STATS.policy_forward_ms << '\n'
              << "policy_forward_calls=" << Common::STATS.policy_forward_calls
              << '\n'
              << "agent_collisions=" << agent_collisions << '\n'
              << "obstacle_collisions=" << obstacle_collisions << '\n'
              << "repeated_candidates=" << repeated_candidates << '\n'
              << "repeat_retry_attempts=" << repeat_retry_attempts << '\n'
              << "repeat_constraints=" << repeat_constraints << '\n'
              << "repeat_detection_steps=" << repeat_detection_steps << '\n'
              << "repeat_escapes=" << repeat_escapes << '\n'
              << "repeat_unescaped=" << repeat_unescaped << '\n'
              << "argmax_conflict_steps=" << argmax_conflict_steps << '\n'
              << "argmax_vertex_conflict_groups="
              << argmax_vertex_conflict_groups << '\n'
              << "argmax_swap_conflicts=" << argmax_swap_conflicts << '\n'
              << "argmax_conflicting_actions="
              << argmax_conflicting_actions << '\n'
              << "pibt_shield_steps=" << pibt_shield_steps << '\n'
              << "pibt_shielded_actions=" << pibt_shielded_actions << '\n';
    std::cout << "fallback_active=" << (fallback_ever_active ? 1 : 0) << '\n'
              << "fallback_activation_step=" << fallback_activation_step
              << '\n'
              << "fallback_activations=" << fallback_activations << '\n'
              << "fallback_steps=" << fallback_steps << '\n';

    if (!args.output_prefix.empty()) {
      std::ofstream summary(args.output_prefix + ".summary.json");
      if (!summary) {
        throw std::runtime_error("failed to open summary JSON");
      }
      summary << "{\n"
              << "  \"schema\": \"dmm-native-result/v1\",\n"
              << "  \"status\": \""
              << (solved ? "solved" : "no_solution") << "\",\n"
              << "  \"solved\": " << (solved ? "true" : "false") << ",\n"
              << "  \"num_agents\": " << ins.N << ",\n"
              << "  \"episode_steps\": " << episode_steps << ",\n"
              << "  \"reached_agents\": " << reached << ",\n"
              << "  \"isr\": " << static_cast<double>(reached) / ins.N
              << ",\n"
              << "  \"soc\": " << (solved ? soc : -1) << ",\n"
              << "  \"makespan\": " << (solved ? makespan : -1) << ",\n"
              << "  \"runtime_sec\": " << runtime_sec << ",\n"
              << "  \"policy_forward_ms\": "
              << Common::STATS.policy_forward_ms << ",\n"
              << "  \"policy_forward_calls\": "
              << Common::STATS.policy_forward_calls << ",\n"
              << "  \"agent_collisions\": " << agent_collisions << ",\n"
              << "  \"obstacle_collisions\": " << obstacle_collisions
              << ",\n"
              << "  \"repeat_escapes\": " << repeat_escapes << ",\n"
              << "  \"repeat_unescaped\": " << repeat_unescaped << ",\n"
              << "  \"pibt_shield_steps\": " << pibt_shield_steps << ",\n"
              << "  \"pibt_shielded_actions\": "
              << pibt_shielded_actions << ",\n"
              << "  \"trajectory\": \"" << args.output_prefix
              << ".trajectory.tsv\",\n"
              << "  \"decisions\": \"" << args.output_prefix
              << ".decisions.tsv\"\n"
              << "}\n";
    }

    if (args.diagnostics && !solved) {
      std::vector<int> occupant(ins.G->size(), -1);
      std::vector<int> unresolved;
      for (int i = 0; i < static_cast<int>(ins.N); ++i) {
        occupant[current[i]->id] = i;
        if (current[i] != ins.goals[i]) unresolved.push_back(i);
      }
      int last_other_arrival = 0;
      for (int i = 0; i < static_cast<int>(ins.N); ++i) {
        if (std::find(unresolved.begin(), unresolved.end(), i) ==
            unresolved.end()) {
          last_other_arrival =
              std::max(last_other_arrival, solve_time[i] + 1);
        }
      }
      std::cout << "diag_unresolved_count=" << unresolved.size() << '\n'
                << "diag_last_other_arrival_step="
                << last_other_arrival << '\n'
                << "diag_alone_tail_steps="
                << episode_steps - last_other_arrival << '\n';

      const int recorded_steps =
          std::min(episode_steps, args.trace_window);
      const int first_step = episode_steps - recorded_steps;
      for (const int agent : unresolved) {
        std::array<int, 5> preferred_counts = {};
        std::array<int, 5> executed_counts = {};
        int stationary = 0;
        int overridden = 0;
        int blocked = 0;
        int blocked_by_goal = 0;
        int min_distance = distances.get(agent, current[agent]);
        int max_distance = min_distance;
        std::unordered_map<int, int> vertex_counts;
        std::unordered_map<int, int> blocker_counts;
        for (int step = first_step; step < episode_steps; ++step) {
          const size_t offset =
              static_cast<size_t>(step % args.trace_window) * ins.N + agent;
          const int pre_id = trace_pre[offset];
          const int post_id = trace_post[offset];
          const int preferred = trace_preferred[offset];
          const int executed = trace_executed[offset];
          if (preferred >= 0 && preferred < 5) ++preferred_counts[preferred];
          if (executed >= 0 && executed < 5) ++executed_counts[executed];
          if (pre_id == post_id) ++stationary;
          if (preferred != executed) ++overridden;
          ++vertex_counts[post_id];
          const int distance = distances.get(agent, post_id);
          min_distance = std::min(min_distance, distance);
          max_distance = std::max(max_distance, distance);

          if (preferred != 0 && preferred != executed) {
            const Vertex* pre = ins.G->V[pre_id];
            const int px = pre->x +
                (preferred == 4 ? 1 : preferred == 3 ? -1 : 0);
            const int py = pre->y +
                (preferred == 2 ? 1 : preferred == 1 ? -1 : 0);
            if (px >= 0 && px < ins.G->width &&
                py >= 0 && py < ins.G->height) {
              const Vertex* desired =
                  ins.G->U[ins.G->width * py + px];
              if (desired != nullptr) {
                const size_t step_base =
                    static_cast<size_t>(step % args.trace_window) * ins.N;
                for (int other = 0; other < static_cast<int>(ins.N);
                     ++other) {
                  if (other == agent) continue;
                  if (trace_pre[step_base + other] != desired->id) continue;
                  ++blocked;
                  ++blocker_counts[other];
                  if (desired == ins.goals[other]) ++blocked_by_goal;
                  break;
                }
              }
            }
          }
        }

        auto counts_text = [](const std::array<int, 5>& counts) {
          std::string text;
          for (int action = 0; action < 5; ++action) {
            if (action) text += ",";
            text += std::to_string(counts[action]);
          }
          return text;
        };
        std::vector<std::pair<int, int>> blockers(
            blocker_counts.begin(), blocker_counts.end());
        std::sort(blockers.begin(), blockers.end(),
                  [](const auto& a, const auto& b) {
                    return a.second > b.second;
                  });
        std::string blocker_text;
        for (size_t k = 0; k < std::min<size_t>(blockers.size(), 5); ++k) {
          if (k) blocker_text += ",";
          const int other = blockers[k].first;
          blocker_text += std::to_string(other) + ":" +
              std::to_string(blockers[k].second) + ":" +
              (current[other] == ins.goals[other] ? "goal" : "active");
        }

        std::cout << "diag_agent=" << agent << '\n'
                  << "diag_start_xy=" << ins.starts[agent]->x << ","
                  << ins.starts[agent]->y << '\n'
                  << "diag_final_xy=" << current[agent]->x << ","
                  << current[agent]->y << '\n'
                  << "diag_goal_xy=" << ins.goals[agent]->x << ","
                  << ins.goals[agent]->y << '\n'
                  << "diag_final_shortest_distance="
                  << distances.get(agent, current[agent]) << '\n'
                  << "diag_trace_min_distance=" << min_distance << '\n'
                  << "diag_trace_max_distance=" << max_distance << '\n'
                  << "diag_last_move_step=" << last_move_step[agent] << '\n'
                  << "diag_trace_steps=" << recorded_steps << '\n'
                  << "diag_trace_unique_vertices=" << vertex_counts.size()
                  << '\n'
                  << "diag_trace_stationary_steps=" << stationary << '\n'
                  << "diag_trace_overridden_steps=" << overridden << '\n'
                  << "diag_trace_blocked_steps=" << blocked << '\n'
                  << "diag_trace_blocked_by_goal_steps="
                  << blocked_by_goal << '\n'
                  << "diag_preferred_action_counts="
                  << counts_text(preferred_counts) << '\n'
                  << "diag_executed_action_counts="
                  << counts_text(executed_counts) << '\n'
                  << "diag_top_blockers=" << blocker_text << '\n';

        static const std::array<int, 5> dx = {0, 0, 0, -1, 1};
        static const std::array<int, 5> dy = {0, -1, 1, 0, 0};
        std::string surroundings;
        for (int action = 0; action < 5; ++action) {
          if (action) surroundings += ";";
          const int x = current[agent]->x + dx[action];
          const int y = current[agent]->y + dy[action];
          surroundings += std::to_string(action) + ":";
          if (x < 0 || x >= ins.G->width ||
              y < 0 || y >= ins.G->height ||
              ins.G->U[ins.G->width * y + x] == nullptr) {
            surroundings += "obstacle";
            continue;
          }
          const Vertex* vertex = ins.G->U[ins.G->width * y + x];
          surroundings += std::to_string(x) + "," + std::to_string(y) +
              ",d=" + std::to_string(distances.get(agent, vertex));
          const int other = occupant[vertex->id];
          if (other >= 0 && other != agent) {
            surroundings += ",occ=" + std::to_string(other) +
                (current[other] == ins.goals[other] ? ":goal" : ":active");
          }
        }
        std::cout << "diag_final_surroundings=" << surroundings << '\n';
      }
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 2;
  }
}

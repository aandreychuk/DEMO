#include "../include/policy.hpp"
#include "lc_mapf_standalone.cpp"  // observation tokenizer and optional LC-MAPF wrapper
#include "mapf_gpt_standalone.cpp"  // optional MAPF-GPT wrapper used by shared policy code
#include "dmm_standalone.cpp"       // DMM TorchScript/AOTI wrapper

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <torch/nn/functional.h>
#include <torch/torch.h>
#include <filesystem>

using namespace Common;

// Forward decl — defined later in this file. Needed because
// build_preferences_from_ensemble_variant (added below) uses it before
// its definition.
int get_policy_action_index_from_vertex(Vertex* v_from, Vertex* v_to);

int AgentPolicy::OBSERVATION_RAD = 5;
bool AgentPolicy::USE_COMMUNICATION_RADIUS = false;
int AgentPolicy::COMMUNICATION_RADIUS = 7;
float AgentPolicy::SAMPLING_TEMPERTURE = 1.0;
AgentPolicy::SamplingStrategy AgentPolicy::SAMPLING_STRATEGY =
    SamplingStrategy::Deterministic;
int AgentPolicy::NN_GATE_MAX_AMBIGUOUS = -1;  // gate disabled by default
constexpr float PI = 3.14;

namespace {
std::mutex hmagat_callback_mutex;
HMAGATInferenceCallback hmagat_callback = nullptr;
}  // namespace

extern "C" int set_hmagat_inference_callback(void* callback)
{
  std::lock_guard<std::mutex> lock(hmagat_callback_mutex);
  hmagat_callback = reinterpret_cast<HMAGATInferenceCallback>(callback);
  return hmagat_callback == nullptr ? 1 : 0;
}

extern "C" int clear_hmagat_inference_callback()
{
  std::lock_guard<std::mutex> lock(hmagat_callback_mutex);
  hmagat_callback = nullptr;
  return 0;
}

bool hmagat_inference_available()
{
  std::lock_guard<std::mutex> lock(hmagat_callback_mutex);
  return hmagat_callback != nullptr;
}

int run_hmagat_inference(const int* positions_xy, int num_agents,
                         float* action_scores, int num_actions)
{
  HMAGATInferenceCallback callback = nullptr;
  {
    std::lock_guard<std::mutex> lock(hmagat_callback_mutex);
    callback = hmagat_callback;
  }
  if (callback == nullptr) return 1;
  return callback(positions_xy, num_agents, action_scores, num_actions);
}

// Ensemble use_model: true if any member is present and its model path is valid.
static bool ensemble_has_model(const PolicyConfig& config)
{
  if (config.policy_type != "ensemble") return false;
  for (const auto& m : config.ensemble_members) {
    if (m == "magat" && !config.model_filepath.empty()
        && std::filesystem::exists(config.model_filepath)) return true;
    if (m == "lc_mapf" && !config.lc_mapf_model_path.empty()) return true;
    if (m == "mapf_gpt" && !config.mapf_gpt_model_path.empty()) return true;
    if (m == "dmm" && !config.dmm_model_path.empty()) return true;
  }
  return false;
}

// Helpers: is sub-policy X requested by this ensemble config?
static bool ensemble_wants(const PolicyConfig& config, const std::string& name)
{
  for (const auto& m : config.ensemble_members) if (m == name) return true;
  return false;
}

AgentPolicy::AgentPolicy(const Instance* _ins, DistTable* _D, const PolicyConfig& config, int seed)
    : ins(_ins),
      MT(seed),
      rrd(0, 1),
      N(ins->N),
      V_size(ins->G->size()),
      D(_D),
      occupied_now(V_size, NO_AGENT),
      use_model(config.policy_type == "magat"
                    ? (!config.model_filepath.empty()
                       && std::filesystem::exists(config.model_filepath))
                    : (config.policy_type == "hmagat"
                       ? hmagat_inference_available()
                       : (config.policy_type == "lc_mapf" || config.policy_type == "mapf_gpt"
                       || config.policy_type == "dmm"
                       || (config.policy_type == "ensemble" && ensemble_has_model(config))))),
      policy_type_(config.policy_type),
      fov_size((OBSERVATION_RAD + 1) * 2 + 1),
      inputs(2),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU),
      node_feature_(torch::zeros({N, 4, (OBSERVATION_RAD + 1) * 2 + 1, (OBSERVATION_RAD + 1) * 2 + 1}, torch::kFloat32)),
      global_guide(ins, D, Common::DEADLINE, seed),
      preferences(N, std::vector<std::pair<Vertex*, ActionCost>>(5)),
      ensemble_members_(config.ensemble_members)
{
  ensemble_policy_ref_ = config.ensemble_policy_ref;
  info(1, "AgentPolicy ctor: policy_type=", policy_type_,
       ", use_model=", use_model ? "true" : "false");

  // Determine which sub-policies need init. Each can be the sole policy
  // (single-type loops) OR a member of the ensemble.
  const bool init_lc_mapf =
      use_model && ((policy_type_ == "lc_mapf")
                    || (policy_type_ == "ensemble" && ensemble_wants(config, "lc_mapf")
                        && !config.lc_mapf_model_path.empty()));
  const bool init_mapf_gpt =
      use_model && ((policy_type_ == "mapf_gpt")
                    || (policy_type_ == "ensemble" && ensemble_wants(config, "mapf_gpt")
                        && !config.mapf_gpt_model_path.empty()));
  const bool init_dmm =
      use_model && ((policy_type_ == "dmm")
                    || (policy_type_ == "ensemble" && ensemble_wants(config, "dmm")
                        && !config.dmm_model_path.empty()));
  const bool init_magat =
      use_model && ((policy_type_ == "magat")
                    || (policy_type_ == "ensemble" && ensemble_wants(config, "magat")
                        && !config.model_filepath.empty()
                        && std::filesystem::exists(config.model_filepath)));

  // LC-MAPF observation generator is shared by both `lc_mapf` and `dmm`
  // policies (DMM consumes the identical observation format).
  if (init_lc_mapf || init_dmm) {
    info(1, "Initializing LC-MAPF observation generator (shared by lc_mapf/dmm)");

    auto&& W = ins->G->width;
    auto&& H = ins->G->height;
    std::vector<std::vector<int>> grid(H, std::vector<int>(W, 0));
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const auto u = ins->G->U[W * y + x];
        grid[y][x] = (u == nullptr) ? 1 : 0;
      }
    }

    constexpr int LC_MAPF_NUM_AGENTS = 13;
    InputParameters cfg(/*cost2go_value_limit=*/20,
                        /*num_agents=*/LC_MAPF_NUM_AGENTS,
                        /*num_previous_actions=*/5,
                        /*context_size=*/256,
                        /*obs_radius=*/5,
                        /*agents_radius=*/5,
                        /*task_type_id=*/-1);

    lc_obs_gen = std::make_unique<ObservationGenerator>(grid, cfg);

    std::vector<std::pair<int, int>> positions(N), goals(N);
    for (int i = 0; i < N; ++i) {
      positions[i] = {ins->starts[i]->y, ins->starts[i]->x};
      goals[i]     = {ins->goals[i]->y,  ins->goals[i]->x};
    }
    lc_obs_gen->create_agents(positions, goals);

    lc_initialized = true;
  }

  if (init_lc_mapf) {
    lc_model = std::make_unique<lc_mapf::LCMAPFTorchscriptModel>(
        config.lc_mapf_model_path, config.lc_mapf_device);
    info(1, "LC-MAPF model loaded");
  }

  if (init_mapf_gpt) {
    info(1, "Initializing MAPF-GPT-DDG C++ core");

    auto&& W = ins->G->width;
    auto&& H = ins->G->height;
    std::vector<std::vector<int>> grid(H, std::vector<int>(W, 0));
    for (int y = 0; y < H; ++y) {
      for (int x = 0; x < W; ++x) {
        const auto u = ins->G->U[W * y + x];
        grid[y][x] = (u == nullptr) ? 1 : 0;
      }
    }

    constexpr int MG_NUM_AGENTS = 13;
    mapf_gpt::InputParameters mg_cfg(/*cost2go_value_limit=*/20,
                                     /*num_agents=*/MG_NUM_AGENTS,
                                     /*num_previous_actions=*/5,
                                     /*context_size=*/256,
                                     /*obs_radius=*/5,
                                     /*agents_radius=*/5,
                                     /*task_type_id=*/-1);

    mg_obs_gen = std::make_unique<mapf_gpt::ObservationGenerator>(grid, mg_cfg);

    std::vector<std::pair<int, int>> positions(N), goals(N);
    for (int i = 0; i < N; ++i) {
      positions[i] = {ins->starts[i]->y, ins->starts[i]->x};
      goals[i]     = {ins->goals[i]->y,  ins->goals[i]->x};
    }
    mg_obs_gen->create_agents(positions, goals);

    mg_model = std::make_unique<mapf_gpt::MAPFGPTTorchscriptModel>(
        config.mapf_gpt_model_path, config.mapf_gpt_device);

    mg_initialized = true;
    info(1, "MAPF-GPT-DDG C++ core initialized");
  }

  if (init_dmm) {
    info(1, "Initializing DMM model (sharing LC-MAPF observation generator)");
    dmm_model = std::make_unique<dmm::DMMTorchscriptModel>(
        config.dmm_model_path, config.dmm_device);
    dmm_initialized = true;
    info(1, "DMM model loaded");
  }

  if (init_magat) {
    // MAGAT+ policy initialization
    torch::jit::getProfilingMode() = false;

    // Load per-instance model copy for thread-safe parallel execution
    info(1, "Loading per-instance MAGAT+ model copy from: ", config.model_filepath);
    instance_model = std::make_unique<torch::jit::script::Module>(torch::jit::load(config.model_filepath));
    instance_model->to(device);
    instance_model->eval();
    info(1, "Per-instance MAGAT+ model loaded successfully");

    // Warmup inference — only for single-type MAGAT loops. For ensemble loops,
    // warmup happens lazily on first set_preferences call.
    if (policy_type_ == "magat") {
      set_preferences_learned(ins->starts);
      info(1, "finish first inference");
      MODEL_LOAD_MS = 0;
    }
  }

  // space utilization optimization
  global_guide.construct();
}

AgentPolicy::~AgentPolicy() {}

// Build preferences[] from cached entry. Pinned agents get a HUGE boost on
// their pinned action (funcPIBT will choose it first). Free agents use
// LC-MAPF priorities; on variant K>0, one free agent is rotated to its
// rank-1 action (next-most-probable per LC-MAPF) — this produces a different
// child config on each scout revisit without changing pinned consensus.
//
// Caller holds nothing; entry_mtx is acquired internally.
void AgentPolicy::build_preferences_from_ensemble_variant(
    const Config& Q, EnsembleCacheEntry& entry)
{
  // Snapshot variant index + bump for next caller.
  int variant_idx;
  {
    std::lock_guard<std::mutex> lock(entry.entry_mtx);
    variant_idx = entry.next_variant_idx;
    entry.next_variant_idx++;
  }

  // For variant K (K > 0): pick free agent (K-1) mod n_free and rotate its
  // priority so its rank-1 action is boosted instead of rank-0.
  int variant_agent = -1;
  if (variant_idx > 0 && !entry.free_agents.empty()) {
    variant_agent = entry.free_agents[(variant_idx - 1) % entry.free_agents.size()];
  }
  const int variant_rank = (variant_idx > 0)
      ? (1 + static_cast<int>((variant_idx - 1) / std::max<size_t>(1, entry.free_agents.size())))
      : 0;
  // (Beyond rank 4, rotation wraps to rank 1 of next agent → handled by free_agents cycling.)

  auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
    const int v_idx = get_policy_action_index_from_vertex(v_from, v_to);
    int boosted_action;
    float v_val;
    if (entry.pinned_action[i] >= 0) {
      // PINNED: strong boost on consensus action. Pinned ALWAYS uses
      // lc_mapf_probs[i] as the underlying signal so other actions still
      // ordered sensibly if funcPIBT rejects the pinned one.
      boosted_action = entry.pinned_action[i];
      v_val = -entry.lc_mapf_probs[i * 5 + v_idx];
    } else {
      // FREE: LC-MAPF distribution. If this agent is the variant-rotated
      // agent, boost its rank-variant_rank action; else rank-0 (top).
      const int rank = (i == variant_agent)
          ? std::min(4, variant_rank)
          : 0;
      boosted_action = entry.lc_mapf_sorted[i][rank];
      v_val = -entry.lc_mapf_probs[i * 5 + v_idx];
    }
    // Strong boost so funcPIBT picks the chosen action first.
    if (v_idx == boosted_action) v_val -= 2.0f;
    if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
      v_val = v_val / SAMPLING_TEMPERTURE - rng.get();
    }
    auto gg = global_guide.get(i, v_from, v_to);
    if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
      return std::make_pair(v_to, std::make_tuple(gg, (float)D->get(i, v_to), v_val));
    } else {
      return std::make_pair(v_to, std::make_tuple(gg, v_val, rrd(MT)));
    }
  };
  for (int i = 0; i < N; ++i) {
    const auto u = Q[i];
    const auto K = u->neighbor.size();
    for (size_t k = 0; k <= K; ++k) {
      preferences[i][k] = get_cost(i, u, u->actions[k]);
    }
    std::sort(preferences[i].begin(), preferences[i].begin() + K + 1,
              [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  }
}

void AgentPolicy::set_preferences(const Config& Q_from,
                                  const std::set<int>& default_policy_agents,
                                  const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>>* action_history)
{
  Common::STATS.policy_setpref_calls += 1;

  // Scout loop path: if this policy has a pointer to an ensemble loop's
  // policy AND that loop has cached a forward for Q_from, use the cached
  // combined distribution with variant-bumped ranks (avoids NN call AND
  // produces a "next most probable" child per the ensemble's preferences).
  if (ensemble_policy_ref_ != nullptr) {
    EnsembleCacheEntry *entry = ensemble_policy_ref_->get_ensemble_cache(Q_from);
    if (entry != nullptr) {
      build_preferences_from_ensemble_variant(Q_from, *entry);
      if (!default_policy_agents.empty())
        set_preferences_naive(Q_from, default_policy_agents);
      return;
    }
    // Cache miss → fall through to local policy (PIBT-only for scout = naive).
  }

  if (use_model) {
    // NN gate: run the cheap naive pass first, measure how many agents
    // have a tied best / second-best cost tuple. If this ambiguity count
    // is ≤ NN_GATE_MAX_AMBIGUOUS, the naive preferences are already
    // "confident enough" and the NN forward pass would almost certainly
    // reorder nothing meaningful — skip it. Saves ~70 % of NN compute
    // on typical MovingAI configurations where most agents have a clear
    // distance-minimizing move at any given branching point.
    if (NN_GATE_MAX_AMBIGUOUS >= 0) {
      set_preferences_naive(Q_from);
      int ambiguous = 0;
      for (int i = 0; i < N; ++i) {
        const auto K = Q_from[i]->neighbor.size();
        if (K == 0) continue;
        // preferences[i] is sorted ascending by cost-tuple; tie = equal.
        if (std::get<1>(preferences[i][0]) == std::get<1>(preferences[i][1])) {
          ++ambiguous;
          if (ambiguous > NN_GATE_MAX_AMBIGUOUS) break;
        }
      }
      if (ambiguous <= NN_GATE_MAX_AMBIGUOUS) {
        // Gate hit — keep the naive preferences that are already set.
        if (!default_policy_agents.empty())
          set_preferences_naive(Q_from, default_policy_agents);
        return;
      }
    }
    set_preferences_learned(Q_from, action_history);
    if (!default_policy_agents.empty())
      set_preferences_naive(Q_from, default_policy_agents);
  } else {
    set_preferences_naive(Q_from);
  }
}

void AgentPolicy::set_preferences_naive(const Config& Q_from,
                                        const std::set<int>& A)
{
  auto get_cost = [&](const int i, const Vertex* u) {
    auto gg = global_guide.get(i, Q_from[i], u);
    return std::make_tuple(gg, D->get(i, u), rrd(MT));
  };

  auto set = [&](const int i) {
    const auto K = Q_from[i]->neighbor.size();

    // set candidate actions
    for (size_t k = 0; k <= K; ++k) {
      auto u = Q_from[i]->actions[k];
      preferences[i][k] = std::make_pair(u, get_cost(i, u));
    }

    // sort, note: K + 1 is sufficient
    std::sort(
        preferences[i].begin(), preferences[i].begin() + K + 1,
        [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  };

  if (A.empty()) {
    for (int i = 0; i < N; ++i) set(i);
  } else {
    for (auto i : A) set(i);
  }
}

int get_policy_action_index_from_vertex(Vertex* v_from, Vertex* v_to)
{
  if (v_from->x == v_to->x && v_from->y > v_to->y) return 1;  // north
  if (v_from->x == v_to->x && v_from->y < v_to->y) return 2;  // south
  if (v_from->x > v_to->x && v_from->y == v_to->y) return 3;  // west
  if (v_from->x < v_to->x && v_from->y == v_to->y) return 4;  // east
  return 0;                                                   // stay
}

static thread_local int inference_cnt = 0;

void AgentPolicy::set_preferences_learned(const Config& Q,
  const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>>* action_history)
{
  if (policy_type_ == "lc_mapf") {
    set_preferences_learned_lc_mapf(Q, action_history);
    return;
  }
  if (policy_type_ == "hmagat") {
    set_preferences_learned_hmagat(Q);
    return;
  }
  if (policy_type_ == "mapf_gpt") {
    set_preferences_learned_mapf_gpt(Q, action_history);
    return;
  }
  if (policy_type_ == "dmm") {
    set_preferences_learned_dmm(Q, action_history);
    return;
  }
  if (policy_type_ == "ensemble") {
    set_preferences_learned_ensemble(Q, action_history);
    return;
  }

  // Original MAGAT+ code
  c10::InferenceMode guard(true);

  auto itr = known_config_table.find(Q);
  if (itr == known_config_table.end()) {
    // model inference
    auto t_start = std::chrono::steady_clock::now();
    set_features(Q);

    // Use per-instance model for thread-safe parallel execution
    torch::Tensor actions;
    if (instance_model) {
      actions = instance_model->forward(inputs).toTensor();
    } else {
      throw std::runtime_error("AgentPolicy: MAGAT+ use_model=true but no instance_model loaded");
    }

    auto t_end = std::chrono::steady_clock::now();
    auto dur_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start)
            .count();
    ++inference_cnt;
    Common::STATS.policy_forward_calls += 1;
    Common::STATS.policy_forward_ms += dur_ms;
    actions = actions.to(torch::kCPU);
    itr = std::get<0>(known_config_table.emplace(Q, actions));
  }
  auto&& actions = itr->second;

  auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
    auto v_idx = get_policy_action_index_from_vertex(v_from, v_to);
    auto acc = actions.accessor<float, 2>();
    auto v_val = -acc[i][v_idx];
    if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
      v_val = v_val / SAMPLING_TEMPERTURE - rng.get();
    }
    auto gg = global_guide.get(i, v_from, v_to);

    if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
      return std::make_pair(v_to,
                            std::make_tuple(gg, (float)D->get(i, v_to), v_val));
    } else {
      return std::make_pair(v_to, std::make_tuple(gg, v_val, rrd(MT)));
    }
  };

  // set preference
  for (int i = 0; i < N; ++i) {
    const auto u = Q[i];
    const auto K = u->neighbor.size();
    for (size_t k = 0; k <= K; ++k) {
      preferences[i][k] = get_cost(i, u, u->actions[k]);
    }

    std::sort(
        preferences[i].begin(), preferences[i].begin() + K + 1,
        [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  }
}

void AgentPolicy::set_preferences_learned_hmagat(const Config& Q)
{
  auto itr = hmagat_action_scores_cache.find(Q);
  if (itr == hmagat_action_scores_cache.end()) {
    std::vector<int> positions_xy(static_cast<size_t>(N) * 2);
    for (int i = 0; i < N; ++i) {
      positions_xy[static_cast<size_t>(i) * 2] = Q[i]->x;
      positions_xy[static_cast<size_t>(i) * 2 + 1] = Q[i]->y;
    }

    std::vector<float> action_scores(static_cast<size_t>(N) * 5, 0.0f);
    const auto started = std::chrono::steady_clock::now();
    const int status =
        run_hmagat_inference(positions_xy.data(), N, action_scores.data(), 5);
    const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - started)
                                .count();
    if (status != 0) {
      throw std::runtime_error("AgentPolicy: HMAGAT callback failed with code " +
                               std::to_string(status));
    }
    Common::STATS.policy_forward_calls += 1;
    Common::STATS.policy_forward_ms += elapsed_ms;
    itr = std::get<0>(
        hmagat_action_scores_cache.emplace(Q, std::move(action_scores)));
  }

  const auto& scores = itr->second;
  auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
    const int action = get_policy_action_index_from_vertex(v_from, v_to);
    float value = -scores[static_cast<size_t>(i) * 5 + action];
    if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
      value = value / SAMPLING_TEMPERTURE - rng.get();
    }
    const auto guide = global_guide.get(i, v_from, v_to);
    if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
      return std::make_pair(
          v_to, std::make_tuple(guide, static_cast<float>(D->get(i, v_to)),
                                value));
    }
    return std::make_pair(v_to, std::make_tuple(guide, value, rrd(MT)));
  };

  for (int i = 0; i < N; ++i) {
    const auto from = Q[i];
    const auto neighbors = from->neighbor.size();
    for (size_t k = 0; k <= neighbors; ++k) {
      preferences[i][k] = get_cost(i, from, from->actions[k]);
    }
    std::sort(preferences[i].begin(),
              preferences[i].begin() + neighbors + 1,
              [&](auto&& lhs, auto&& rhs) {
                return std::get<1>(lhs) < std::get<1>(rhs);
              });
  }
}

void AgentPolicy::set_features(const Config& Q)
{
  c10::InferenceMode guard(true);

  const float d_invalid = 2 * OBSERVATION_RAD;

  // for edge construction
  for (int i = 0; i < N; ++i) occupied_now[Q[i]->id] = i;

  node_feature_.zero_();
  auto X = node_feature_.accessor<float, 4>();

  int edge_ptr = 0;
  std::vector<std::pair<int, int>> vec_edge_index;
  std::vector<std::tuple<int, int, int>> vec_edge_attr;
  std::vector<std::tuple<int, int, int, int>> mean_vec_edge_attr(
      N, std::make_tuple(0, 0, 0, 0));
  auto add_edge = [&](int src, int dst, float x_rel = 0, float y_rel = 0) {
    if (src == dst) return;
    vec_edge_index.emplace_back(src, dst);
    vec_edge_attr.emplace_back(y_rel, x_rel, std::abs(x_rel) + std::abs(y_rel));
    mean_vec_edge_attr[dst] =
        std::make_tuple(std::get<0>(mean_vec_edge_attr[dst]) + y_rel,
                        std::get<1>(mean_vec_edge_attr[dst]) + x_rel,
                        std::get<2>(mean_vec_edge_attr[dst]) + std::abs(x_rel) +
                            std::abs(y_rel),
                        std::get<3>(mean_vec_edge_attr[dst]) + 1);
    ++edge_ptr;
  };

  auto&& W = ins->G->width;
  auto&& H = ins->G->height;

  // feature and edge_index construction
  for (int i = 0; i < N; ++i) {
    const auto v_i = Q[i];
    const auto d_base = D->get(i, v_i);

    for (auto y_l = 0; y_l < fov_size; ++y_l) {
      const auto y_g = y_l + v_i->y - OBSERVATION_RAD - 1;
      for (auto x_l = 0; x_l < fov_size; ++x_l) {
        const auto x_g = x_l + v_i->x - OBSERVATION_RAD - 1;

        X[i][0][y_l][x_l] = 0;
        X[i][1][y_l][x_l] = 0;
        X[i][2][y_l][x_l] = 0;
        X[i][3][y_l][x_l] = 1.0f;

        if (x_l == 0 || x_l == fov_size - 1 || y_l == 0 ||
            y_l == fov_size - 1) {
          X[i][3][y_l][x_l] = 0;
        } else if (0 <= x_g && x_g < W && 0 <= y_g && y_g < H) {
          const auto u = ins->G->U[W * y_g + x_g];
          if (u != nullptr) {
            X[i][0][y_l][x_l] = 0;

            const auto j = occupied_now[u->id];
            if (j != NO_AGENT) {
              const auto pos_diff_x = v_i->x - u->x;
              const auto pos_diff_y = v_i->y - u->y;
              if (!USE_COMMUNICATION_RADIUS ||
                  (pos_diff_x * pos_diff_x + pos_diff_y * pos_diff_y <=
                   COMMUNICATION_RADIUS * COMMUNICATION_RADIUS)) {
                add_edge(i, j, pos_diff_x, pos_diff_y);
              }
              X[i][1][y_l][x_l] = 1;
            }

            X[i][3][y_l][x_l] =
                std::max(std::min((D->get(i, u) - d_base) / d_invalid, 1.0f),
                         -1.0f);
          } else {
            X[i][0][y_l][x_l] = 1;
          }
        } else if (((x_g == -1 || x_g == W) && -1 <= y_g && y_g <= H) ||
                   ((y_g == -1 || y_g == H) && -1 <= x_g && x_g <= W)) {
          X[i][0][y_l][x_l] = 1;
        }
      }
    }

    if (USE_COMMUNICATION_RADIUS) {
      for (auto y_l = 0; y_l < 2 * (COMMUNICATION_RADIUS - OBSERVATION_RAD); ++y_l) {
        auto y_g = y_l + v_i->y - COMMUNICATION_RADIUS - 1;
        if (y_l >= COMMUNICATION_RADIUS - OBSERVATION_RAD) {
          y_g += 2 * OBSERVATION_RAD + 1;
        }
        for (auto x_l = 0;
             x_l < 2 * (COMMUNICATION_RADIUS - OBSERVATION_RAD); ++x_l) {
          auto x_g = x_l + v_i->x - COMMUNICATION_RADIUS - 1;
          if (x_l >= COMMUNICATION_RADIUS - OBSERVATION_RAD) {
            x_g += 2 * OBSERVATION_RAD + 1;
          }

          if (0 <= x_g && x_g < W && 0 <= y_g && y_g < H) {
            const auto u = ins->G->U[W * y_g + x_g];
            if (u != nullptr) {
              const auto j = occupied_now[u->id];
              if (j != NO_AGENT) {
                const auto pos_diff_x = v_i->x - u->x;
                const auto pos_diff_y = v_i->y - u->y;
                if (pos_diff_x * pos_diff_x + pos_diff_y * pos_diff_y <=
                    COMMUNICATION_RADIUS * COMMUNICATION_RADIUS) {
                  add_edge(i, j, pos_diff_x, pos_diff_y);
                }
              }
            }
          }
        }
      }
    }

    auto&& r = OBSERVATION_RAD + 1;
    const auto d_x = ins->goals[i]->x - v_i->x;
    const auto d_y = ins->goals[i]->y - v_i->y;
    auto p_x = d_x;
    auto p_y = d_y;
    if (std::abs(d_x) > r || std::abs(d_y) > r) {
      auto angle = atan2(d_y, d_x);
      if ((angle >= PI / 4 && angle <= PI * 3 / 4) ||
          (angle >= -PI * 3 / 4 && angle <= -PI / 4)) {
        p_x = (d_y == 0) ? r * (d_x < 0 ? -1 : 1)
                         : std::round(d_x * ((float)r / d_y));
        p_y = r * (d_y < 0 ? -1 : 1);
      } else {
        p_x = r * (d_x < 0 ? -1 : 1);
        p_y = (d_x == 0) ? r * (d_y < 0 ? -1 : 1)
                         : std::round(d_y * ((float)r / d_x));
      }
    }
    X[i][2][p_y + r][p_x + r] = 1;
  }

  for (int i = 0; i < N; ++i) {
    auto&& edge_attr_tuple_i = mean_vec_edge_attr[i];
    auto num_edges = std::get<3>(edge_attr_tuple_i);
    if (num_edges == 0) {
      continue;
    }
    vec_edge_index.emplace_back(i, i);
    vec_edge_attr.emplace_back(std::get<0>(edge_attr_tuple_i) / num_edges,
                               std::get<1>(edge_attr_tuple_i) / num_edges,
                               std::get<2>(edge_attr_tuple_i) / num_edges);
    ++edge_ptr;
  }

  torch::Tensor edge_index = torch::empty({2, edge_ptr}, torch::kInt64);
  torch::Tensor edge_attr = torch::empty({edge_ptr, 3}, torch::kFloat32);
  auto edge_index_acc = edge_index.accessor<int64_t, 2>();
  auto edge_attr_acc = edge_attr.accessor<float, 2>();

  for (int64_t i = 0; i < edge_ptr; ++i) {
    edge_index_acc[0][i] = vec_edge_index[i].first;
    edge_index_acc[1][i] = vec_edge_index[i].second;
    edge_attr_acc[i][0] = std::get<0>(vec_edge_attr[i]);
    edge_attr_acc[i][1] = std::get<1>(vec_edge_attr[i]);
    edge_attr_acc[i][2] = std::get<2>(vec_edge_attr[i]);
  }

  auto kwargs = torch::Dict<std::string, torch::Tensor>();
  kwargs.insert("edge_index", edge_index.to(device));
  kwargs.insert("edge_attr", edge_attr.to(device));

  inputs[0] = node_feature_.to(device);
  inputs[1] = kwargs;

  // cleanup
  for (int i = 0; i < N; ++i) occupied_now[Q[i]->id] = NO_AGENT;
}

void AgentPolicy::set_preferences_learned_lc_mapf(const Config& Q,
  const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>>* action_history)
{
  if (!lc_initialized || !lc_obs_gen || !lc_model) {
    info(0, "set_preferences_learned_lc_mapf: LC-MAPF core not initialized, "
             "falling back to naive policy");
    set_preferences_naive(Q);
    return;
  }

  auto itr = (action_history == nullptr) ? lc_mapf_action_logits_cache.find(Q) : lc_mapf_action_logits_cache.end();
  if (itr != lc_mapf_action_logits_cache.end()) {
    const std::vector<float>& action_logits = itr->second;
    auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
      auto v_idx = get_policy_action_index_from_vertex(v_from, v_to);
      auto v_val = -action_logits[i * 5 + v_idx];
      if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
        v_val = v_val / SAMPLING_TEMPERTURE - rng.get();
      }
      auto gg = global_guide.get(i, v_from, v_to);
      if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
        return std::make_pair(v_to, std::make_tuple(gg, (float)D->get(i, v_to), v_val));
      } else {
        return std::make_pair(v_to, std::make_tuple(gg, v_val, rrd(MT)));
      }
    };
    for (int i = 0; i < N; ++i) {
      const auto u = Q[i];
      const auto K = u->neighbor.size();
      for (size_t k = 0; k <= K; ++k) {
        preferences[i][k] = get_cost(i, u, u->actions[k]);
      }
      std::sort(preferences[i].begin(), preferences[i].begin() + K + 1,
                [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
    }
    return;
  }

  auto t_start = std::chrono::steady_clock::now();

  // Phase-by-phase timing for LC-MAPF GPU port investigation.
  // Set LAGAT_LC_PROFILE=1 to print breakdown per call.
  static const bool lc_profile = std::getenv("LAGAT_LC_PROFILE") != nullptr;
  static int lc_profile_call_idx = 0;
  static long long lc_t_pos_us = 0, lc_t_update_us = 0,
      lc_t_gen_obs_us = 0, lc_t_get_chat_us = 0,
      lc_t_int64_us = 0, lc_t_forward_us = 0;
  auto lc_clock_now = []() {
    return std::chrono::steady_clock::now();
  };
  auto lc_t0 = lc_clock_now();

  std::vector<std::pair<int, int>> positions(N), goals(N);
  for (int i = 0; i < N; ++i) {
    positions[i] = {Q[i]->y, Q[i]->x};
    goals[i]     = {ins->goals[i]->y, ins->goals[i]->x};
  }
  auto lc_t1 = lc_clock_now();

  const bool use_history = action_history != nullptr &&
                           action_history->size() == static_cast<size_t>(N);
  if (use_history) {
    lc_obs_gen->update_agents(positions, goals, *action_history);
  } else {
    std::vector<int> actions(N, -1);
    lc_obs_gen->update_agents(positions, goals, actions);
  }
  auto lc_t2 = lc_clock_now();

  auto encoded_obs    = lc_obs_gen->generate_observations();
  auto lc_t3 = lc_clock_now();

  auto agent_chat_ids = lc_obs_gen->get_agents_in_obs();
  auto lc_t4 = lc_clock_now();

  std::vector<std::vector<int64_t>> obs_int64(encoded_obs.size());
  for (size_t i = 0; i < encoded_obs.size(); ++i) {
    obs_int64[i].assign(encoded_obs[i].begin(), encoded_obs[i].end());
  }
  std::vector<std::vector<int64_t>> chat_int64(agent_chat_ids.size());
  for (size_t i = 0; i < agent_chat_ids.size(); ++i) {
    chat_int64[i].assign(agent_chat_ids[i].begin(), agent_chat_ids[i].end());
  }
  auto lc_t5 = lc_clock_now();

  auto probs = lc_model->action_probs_from_vectors_std(obs_int64, chat_int64);
  auto lc_t6 = lc_clock_now();

  if (lc_profile) {
    auto us = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    lc_t_pos_us     += us(lc_t0, lc_t1);
    lc_t_update_us  += us(lc_t1, lc_t2);
    lc_t_gen_obs_us += us(lc_t2, lc_t3);
    lc_t_get_chat_us+= us(lc_t3, lc_t4);
    lc_t_int64_us   += us(lc_t4, lc_t5);
    lc_t_forward_us += us(lc_t5, lc_t6);
    ++lc_profile_call_idx;
    if (lc_profile_call_idx % 50 == 0) {
      info(0, "[lc-profile] calls=", lc_profile_call_idx,
        " sums(ms): pos=", lc_t_pos_us / 1000,
        " update=", lc_t_update_us / 1000,
        " gen_obs=", lc_t_gen_obs_us / 1000,
        " get_chat=", lc_t_get_chat_us / 1000,
        " int64=", lc_t_int64_us / 1000,
        " forward=", lc_t_forward_us / 1000);
    }
  }
  if (probs.size() != static_cast<size_t>(N)) {
    throw std::runtime_error("LC-MAPF returned probs with unexpected batch size");
  }
  if (!probs.empty() && probs.front().size() != 5) {
    throw std::runtime_error("LC-MAPF expected 5 actions per agent");
  }

  std::vector<float> action_logits(N * 5);
  for (int i = 0; i < N; ++i) {
    for (int a = 0; a < 5; ++a) {
      action_logits[i * 5 + a] = probs[static_cast<size_t>(i)][static_cast<size_t>(a)];
    }
  }

  if (action_history == nullptr)
    lc_mapf_action_logits_cache[Q] = std::move(action_logits);

  auto t_end = std::chrono::steady_clock::now();
  auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  ++inference_cnt;
  Common::STATS.policy_forward_calls += 1;
  Common::STATS.policy_forward_ms += dur_ms;

  const std::vector<float>& action_logits_ref =
      (action_history == nullptr) ? lc_mapf_action_logits_cache[Q] : action_logits;

  auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
    auto v_idx = get_policy_action_index_from_vertex(v_from, v_to);
    auto v_val = -action_logits_ref[i * 5 + v_idx];
    if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
      v_val = v_val / SAMPLING_TEMPERTURE - rng.get();
    }
    auto gg = global_guide.get(i, v_from, v_to);

    if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
      return std::make_pair(v_to,
                            std::make_tuple(gg, (float)D->get(i, v_to), v_val));
    } else {
      return std::make_pair(v_to, std::make_tuple(gg, v_val, rrd(MT)));
    }
  };

  for (int i = 0; i < N; ++i) {
    const auto u = Q[i];
    const auto K = u->neighbor.size();
    for (size_t k = 0; k <= K; ++k) {
      preferences[i][k] = get_cost(i, u, u->actions[k]);
    }

    std::sort(
        preferences[i].begin(), preferences[i].begin() + K + 1,
        [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  }
}

void AgentPolicy::set_preferences_learned_mapf_gpt(const Config& Q,
  const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>>* action_history)
{
  if (!mg_initialized || !mg_obs_gen || !mg_model) {
    info(0, "set_preferences_learned_mapf_gpt: MAPF-GPT-DDG core not initialized, "
             "falling back to naive policy");
    set_preferences_naive(Q);
    return;
  }

  auto itr = (action_history == nullptr) ? mapf_gpt_action_logits_cache.find(Q) : mapf_gpt_action_logits_cache.end();
  if (itr != mapf_gpt_action_logits_cache.end()) {
    const std::vector<float>& action_logits = itr->second;
    auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
      auto v_idx = get_policy_action_index_from_vertex(v_from, v_to);
      auto v_val = -action_logits[i * 5 + v_idx];
      if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
        v_val = v_val / SAMPLING_TEMPERTURE - rng.get();
      }
      auto gg = global_guide.get(i, v_from, v_to);
      if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
        return std::make_pair(v_to, std::make_tuple(gg, (float)D->get(i, v_to), v_val));
      } else {
        return std::make_pair(v_to, std::make_tuple(gg, v_val, rrd(MT)));
      }
    };
    for (int i = 0; i < N; ++i) {
      const auto u = Q[i];
      const auto K = u->neighbor.size();
      for (size_t k = 0; k <= K; ++k) {
        preferences[i][k] = get_cost(i, u, u->actions[k]);
      }
      std::sort(preferences[i].begin(), preferences[i].begin() + K + 1,
                [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
    }
    return;
  }

  auto t_start = std::chrono::steady_clock::now();

  std::vector<std::pair<int, int>> positions(N), goals(N);
  for (int i = 0; i < N; ++i) {
    positions[i] = {Q[i]->y, Q[i]->x};
    goals[i]     = {ins->goals[i]->y, ins->goals[i]->x};
  }

  const bool use_history = action_history != nullptr &&
                           action_history->size() == static_cast<size_t>(N);
  if (use_history) {
    mg_obs_gen->update_agents(positions, goals, *action_history);
  } else {
    std::vector<int> actions(N, -1);
    mg_obs_gen->update_agents(positions, goals, actions);
  }

  auto encoded_obs = mg_obs_gen->generate_observations();

  std::vector<std::vector<int64_t>> obs_int64(encoded_obs.size());
  for (size_t i = 0; i < encoded_obs.size(); ++i) {
    obs_int64[i].assign(encoded_obs[i].begin(), encoded_obs[i].end());
  }

  auto probs = mg_model->action_probs_from_vectors_std(obs_int64);
  if (probs.size() != static_cast<size_t>(N)) {
    throw std::runtime_error("MAPF-GPT-DDG returned probs with unexpected batch size");
  }
  if (!probs.empty() && probs.front().size() != 5) {
    throw std::runtime_error("MAPF-GPT-DDG expected 5 actions per agent");
  }

  std::vector<float> action_logits(N * 5);
  for (int i = 0; i < N; ++i) {
    for (int a = 0; a < 5; ++a) {
      action_logits[i * 5 + a] = probs[static_cast<size_t>(i)][static_cast<size_t>(a)];
    }
  }

  if (action_history == nullptr)
    mapf_gpt_action_logits_cache[Q] = std::move(action_logits);

  auto t_end = std::chrono::steady_clock::now();
  auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  ++inference_cnt;
  Common::STATS.policy_forward_calls += 1;
  Common::STATS.policy_forward_ms += dur_ms;

  const std::vector<float>& action_logits_ref =
      (action_history == nullptr) ? mapf_gpt_action_logits_cache[Q] : action_logits;

  auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
    auto v_idx = get_policy_action_index_from_vertex(v_from, v_to);
    auto v_val = -action_logits_ref[i * 5 + v_idx];
    if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
      v_val = v_val / SAMPLING_TEMPERTURE - rng.get();
    }
    auto gg = global_guide.get(i, v_from, v_to);

    if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
      return std::make_pair(v_to,
                            std::make_tuple(gg, (float)D->get(i, v_to), v_val));
    } else {
      return std::make_pair(v_to, std::make_tuple(gg, v_val, rrd(MT)));
    }
  };

  for (int i = 0; i < N; ++i) {
    const auto u = Q[i];
    const auto K = u->neighbor.size();
    for (size_t k = 0; k <= K; ++k) {
      preferences[i][k] = get_cost(i, u, u->actions[k]);
    }

    std::sort(
        preferences[i].begin(), preferences[i].begin() + K + 1,
        [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  }
}

void AgentPolicy::set_preferences_learned_dmm(const Config& Q,
  const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>>* action_history)
{
  if (!dmm_initialized || !lc_obs_gen || !dmm_model) {
    info(0, "set_preferences_learned_dmm: DMM core not initialized, "
             "falling back to naive policy");
    set_preferences_naive(Q);
    return;
  }

  auto t_start = std::chrono::steady_clock::now();

  // Phase-by-phase timing for DMM GPU port investigation.
  // Set LAGAT_LC_PROFILE=1 to print breakdown per call.
  static const bool lc_profile = std::getenv("LAGAT_LC_PROFILE") != nullptr;
  static int lc_profile_call_idx = 0;
  static long long lc_t_pos_us = 0, lc_t_update_us = 0,
      lc_t_gen_obs_us = 0, lc_t_get_chat_us = 0,
      lc_t_int64_us = 0, lc_t_forward_us = 0;
  auto lc_clock_now = []() {
    return std::chrono::steady_clock::now();
  };
  auto lc_t0 = lc_clock_now();

  std::vector<std::pair<int, int>> positions(N), goals(N);
  for (int i = 0; i < N; ++i) {
    positions[i] = {Q[i]->y, Q[i]->x};
    goals[i]     = {ins->goals[i]->y, ins->goals[i]->x};
  }
  auto lc_t1 = lc_clock_now();

  const bool use_history = action_history != nullptr &&
                           action_history->size() == static_cast<size_t>(N);
  if (use_history) {
    lc_obs_gen->update_agents(positions, goals, *action_history);
  } else {
    std::vector<int> actions(N, -1);
    lc_obs_gen->update_agents(positions, goals, actions);
  }
  auto lc_t2 = lc_clock_now();

  auto encoded_obs    = lc_obs_gen->generate_observations();
  auto lc_t3 = lc_clock_now();

  auto agent_chat_ids = lc_obs_gen->get_agents_in_obs();
  auto lc_t4 = lc_clock_now();

  std::vector<std::vector<int64_t>> obs_int64(encoded_obs.size());
  for (size_t i = 0; i < encoded_obs.size(); ++i) {
    obs_int64[i].assign(encoded_obs[i].begin(), encoded_obs[i].end());
  }
  std::vector<std::vector<int64_t>> chat_int64(agent_chat_ids.size());
  for (size_t i = 0; i < agent_chat_ids.size(); ++i) {
    chat_int64[i].assign(agent_chat_ids[i].begin(), agent_chat_ids[i].end());
  }
  std::vector<char> active_mask(static_cast<size_t>(N), false);
  for (int i = 0; i < N; ++i) {
    active_mask[static_cast<size_t>(i)] = Q[i] != ins->goals[i];
  }
  auto lc_t5 = lc_clock_now();

  auto probs = dmm_model->action_probs_from_vectors_std(
      obs_int64, chat_int64, active_mask);
  auto lc_t6 = lc_clock_now();

  if (lc_profile) {
    auto us = [](auto a, auto b) {
      return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };
    lc_t_pos_us     += us(lc_t0, lc_t1);
    lc_t_update_us  += us(lc_t1, lc_t2);
    lc_t_gen_obs_us += us(lc_t2, lc_t3);
    lc_t_get_chat_us+= us(lc_t3, lc_t4);
    lc_t_int64_us   += us(lc_t4, lc_t5);
    lc_t_forward_us += us(lc_t5, lc_t6);
    ++lc_profile_call_idx;
    if (lc_profile_call_idx % 50 == 0) {
      info(0, "[dmm-profile] calls=", lc_profile_call_idx,
        " sums(ms): pos=", lc_t_pos_us / 1000,
        " update=", lc_t_update_us / 1000,
        " gen_obs=", lc_t_gen_obs_us / 1000,
        " get_chat=", lc_t_get_chat_us / 1000,
        " int64=", lc_t_int64_us / 1000,
        " forward=", lc_t_forward_us / 1000);
    }
  }
  if (probs.size() != static_cast<size_t>(N)) {
    throw std::runtime_error("DMM returned probs with unexpected batch size");
  }
  if (!probs.empty() && probs.front().size() != 5) {
    throw std::runtime_error("DMM expected 5 actions per agent");
  }

  std::vector<float> action_logits(N * 5);
  for (int i = 0; i < N; ++i) {
    for (int a = 0; a < 5; ++a) {
      action_logits[i * 5 + a] = probs[static_cast<size_t>(i)][static_cast<size_t>(a)];
    }
  }

  auto t_end = std::chrono::steady_clock::now();
  auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
  ++inference_cnt;
  Common::STATS.policy_forward_calls += 1;
  Common::STATS.policy_forward_ms += dur_ms;

  auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
    auto v_idx = get_policy_action_index_from_vertex(v_from, v_to);
    auto v_val = -action_logits[i * 5 + v_idx];
    if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
      v_val = v_val / SAMPLING_TEMPERTURE - rng.get();
    }
    auto gg = global_guide.get(i, v_from, v_to);

    if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
      return std::make_pair(v_to,
                            std::make_tuple(gg, (float)D->get(i, v_to), v_val));
    } else {
      return std::make_pair(v_to, std::make_tuple(gg, v_val, rrd(MT)));
    }
  };

  for (int i = 0; i < N; ++i) {
    const auto u = Q[i];
    const auto K = u->neighbor.size();
    for (size_t k = 0; k <= K; ++k) {
      preferences[i][k] = get_cost(i, u, u->actions[k]);
    }

    std::sort(
        preferences[i].begin(), preferences[i].begin() + K + 1,
        [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  }
}

std::vector<std::vector<float>> AgentPolicy::get_dmm_action_probabilities(
    const Config& Q,
    const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>>*
        action_history)
{
  if (!dmm_initialized || !lc_obs_gen || !dmm_model) {
    throw std::runtime_error(
        "get_dmm_action_probabilities: DMM core is not initialized");
  }

  const auto t_start = std::chrono::steady_clock::now();
  std::vector<std::pair<int, int>> positions(N), goals(N);
  for (int i = 0; i < N; ++i) {
    positions[i] = {Q[i]->y, Q[i]->x};
    goals[i] = {ins->goals[i]->y, ins->goals[i]->x};
  }

  const bool use_history =
      action_history != nullptr &&
      action_history->size() == static_cast<size_t>(N);
  if (use_history) {
    lc_obs_gen->update_agents(positions, goals, *action_history);
  } else {
    lc_obs_gen->update_agents(positions, goals, std::vector<int>(N, -1));
  }

  const auto encoded_obs = lc_obs_gen->generate_observations();
  const auto agent_chat_ids = lc_obs_gen->get_agents_in_obs();
  std::vector<std::vector<int64_t>> obs_int64(encoded_obs.size());
  std::vector<std::vector<int64_t>> chat_int64(agent_chat_ids.size());
  for (size_t i = 0; i < encoded_obs.size(); ++i) {
    obs_int64[i].assign(encoded_obs[i].begin(), encoded_obs[i].end());
  }
  for (size_t i = 0; i < agent_chat_ids.size(); ++i) {
    chat_int64[i].assign(agent_chat_ids[i].begin(), agent_chat_ids[i].end());
  }
  std::vector<char> active_mask(static_cast<size_t>(N), false);
  for (int i = 0; i < N; ++i) {
    active_mask[static_cast<size_t>(i)] = Q[i] != ins->goals[i];
  }

  auto probs = dmm_model->action_probs_from_vectors_std(
      obs_int64, chat_int64, active_mask);
  if (probs.size() != static_cast<size_t>(N) ||
      (!probs.empty() && probs.front().size() != 5)) {
    throw std::runtime_error(
        "DMM returned action probabilities with an unexpected shape");
  }

  const auto elapsed_ms = std::chrono::duration_cast<
      std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - t_start).count();
  ++inference_cnt;
  Common::STATS.policy_forward_calls += 1;
  Common::STATS.policy_forward_ms += elapsed_ms;
  return probs;
}

void AgentPolicy::get_dmm_encoded_inputs(
    std::vector<std::vector<int>>& observations,
    std::vector<std::vector<int>>& chat) const
{
  if (!lc_obs_gen) {
    throw std::runtime_error("DMM observation generator is not initialized");
  }
  observations = lc_obs_gen->generate_observations();
  chat = lc_obs_gen->get_agents_in_obs();
}

void AgentPolicy::set_dynamic_obstacles(
    const std::vector<std::vector<int>>& blocked_vertex_ids,
    const std::vector<char>& communication_disabled)
{
  if (!lc_obs_gen) return;
  if (blocked_vertex_ids.size() != static_cast<size_t>(N)) {
    throw std::runtime_error(
        "dynamic obstacle rows must match policy agent count");
  }
  std::vector<std::vector<std::pair<int, int>>> cells(N);
  for (int i = 0; i < N; ++i) {
    cells[i].reserve(blocked_vertex_ids[i].size());
    for (const int vertex_id : blocked_vertex_ids[i]) {
      if (vertex_id < 0 || vertex_id >= V_size) continue;
      const auto* vertex = ins->G->V[vertex_id];
      cells[i].push_back({vertex->y, vertex->x});
    }
  }
  lc_obs_gen->set_dynamic_obstacles(cells);
  lc_obs_gen->set_communication_disabled(communication_disabled);
  known_config_table.clear();
  lc_mapf_action_logits_cache.clear();
  ensemble_cache.clear();
}

// ============================================================================
// Ensemble forward — combine N*5 action probabilities from multiple members.
//
// Each member sub-policy runs its own forward and returns (or has cached) an
// N*5 action probability vector. We softmax-normalize each per agent (so all
// members are on the same simplex), then average. The combined vector is
// cached in ensemble_cache[Q]. preferences[] is built from the combined
// distribution (negated values used as priority cost — lower = preferred).
//
// uncertainty = 1 - mean over i of max_a P[i][a]:
//   0.0 → every agent has a confident top action (NNs agree, distribution
//         sharp) — scout NOT interested in this state.
//   0.8 → every agent's top action is barely > uniform (NNs uncertain or
//         disagree) — scout HIGHLY interested.
// ============================================================================

// Helper: in-place softmax on probs[i*5..i*5+4] for each agent.
static void softmax_per_agent(std::vector<float>& probs, int N)
{
  for (int i = 0; i < N; ++i) {
    float* row = probs.data() + i * 5;
    float mx = row[0];
    for (int a = 1; a < 5; ++a) if (row[a] > mx) mx = row[a];
    float sum = 0;
    for (int a = 0; a < 5; ++a) {
      row[a] = std::exp(row[a] - mx);
      sum += row[a];
    }
    if (sum > 0) {
      for (int a = 0; a < 5; ++a) row[a] /= sum;
    } else {
      for (int a = 0; a < 5; ++a) row[a] = 0.2f;  // uniform fallback
    }
  }
}

void AgentPolicy::set_preferences_learned_ensemble(
    const Config& Q,
    const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>>* action_history)
{
  // ---------------------------------------------------------------------------
  // MAGAT-first CASCADE strategy (replaces always-3-NN consensus-pin).
  //
  //   1. Forward MAGAT (~1.5 ms) every call.
  //   2. Per agent i, conf_M[i] = max_a probs_M[i][a].
  //   3. n_uncertain = # agents with conf_M[i] < MAGAT_THRESHOLD.
  //   4. If n_uncertain > 0: forward LC-MAPF (~5 ms). Else: skip it.
  //   5. Per agent decision:
  //        conf_M[i] >= MAGAT_THRESHOLD → pin agent i on argmax MAGAT.
  //        conf_M[i] <  MAGAT_THRESHOLD → free; priorities = LC-MAPF if it ran,
  //                                       else MAGAT (fallback).
  //
  // Perf rationale: when MAGAT is confident on everyone we pay 1.5 ms instead
  // of ~7.7 ms (MAGAT+LC-MAPF+MAPF-GPT). MAPF-GPT is intentionally not called
  // in this 2-tier cascade (loadable but unused).
  // ---------------------------------------------------------------------------
  constexpr float MAGAT_THRESHOLD = 0.6f;

  // Cache lookup: same-Config repeat → reuse cached cascade result, no NN calls.
  // Only cache when action_history is null (stateless: same Q → same answer).
  if (action_history == nullptr) {
    EnsembleCacheEntry* hit = nullptr;
    {
      std::lock_guard<std::mutex> lock(ensemble_cache_mtx);
      auto cached = ensemble_cache.find(Q);
      if (cached != ensemble_cache.end()) {
        hit = cached->second.get();
      }
    }
    if (hit != nullptr) {
      // Ensemble loop's own re-visit uses variant logic too (advances rank
      // for next scout caller).
      build_preferences_from_ensemble_variant(Q, *hit);
      return;
    }
  }

  // ---- Tier 1: MAGAT forward (always, if loaded) ----
  std::vector<float> probs_M;  // N*5, empty if MAGAT not loaded
  if (instance_model) {
    c10::InferenceMode guard(true);
    auto t_start = std::chrono::steady_clock::now();

    set_features(Q);
    torch::Tensor actions_t = instance_model->forward(inputs).toTensor();
    actions_t = actions_t.to(torch::kCPU);

    auto acc = actions_t.accessor<float, 2>();
    probs_M.assign(N * 5, 0.0f);
    for (int i = 0; i < N; ++i) {
      for (int a = 0; a < 5; ++a) probs_M[i * 5 + a] = acc[i][a];
    }
    softmax_per_agent(probs_M, N);

    auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();
    Common::STATS.policy_forward_calls += 1;
    Common::STATS.policy_forward_ms += dur_ms;
  }

  // If MAGAT isn't loaded, the cascade has no leader — try LC-MAPF as the
  // sole tier, else fall back to naive.
  const bool magat_ok = !probs_M.empty();

  // Compute MAGAT confidences + uncertain agent count (only if MAGAT ran).
  std::vector<float> conf_M(N, 0.0f);
  std::vector<int> magat_top(N, -1);
  int n_uncertain = N;  // if MAGAT didn't run, treat all as uncertain
  if (magat_ok) {
    n_uncertain = 0;
    for (int i = 0; i < N; ++i) {
      int top = 0;
      float top_p = probs_M[i * 5 + 0];
      for (int a = 1; a < 5; ++a) {
        if (probs_M[i * 5 + a] > top_p) { top_p = probs_M[i * 5 + a]; top = a; }
      }
      magat_top[i] = top;
      conf_M[i] = top_p;
      if (top_p < MAGAT_THRESHOLD) ++n_uncertain;
    }
  }

  // ---- Tier 2: LC-MAPF forward only if MAGAT left someone uncertain ----
  std::vector<float> probs_L;  // N*5, empty if LC-MAPF didn't run
  const bool want_lc = (n_uncertain > 0) && lc_initialized && lc_model;
  if (want_lc) {
    auto t_start = std::chrono::steady_clock::now();

    std::vector<std::pair<int, int>> positions(N), goals(N);
    for (int i = 0; i < N; ++i) {
      positions[i] = {Q[i]->y, Q[i]->x};
      goals[i]     = {ins->goals[i]->y, ins->goals[i]->x};
    }
    const bool use_history = action_history != nullptr &&
                             action_history->size() == static_cast<size_t>(N);
    if (use_history) {
      lc_obs_gen->update_agents(positions, goals, *action_history);
    } else {
      std::vector<int> actions(N, -1);
      lc_obs_gen->update_agents(positions, goals, actions);
    }
    auto encoded_obs    = lc_obs_gen->generate_observations();
    auto agent_chat_ids = lc_obs_gen->get_agents_in_obs();
    std::vector<std::vector<int64_t>> obs_int64(encoded_obs.size());
    for (size_t i = 0; i < encoded_obs.size(); ++i)
      obs_int64[i].assign(encoded_obs[i].begin(), encoded_obs[i].end());
    std::vector<std::vector<int64_t>> chat_int64(agent_chat_ids.size());
    for (size_t i = 0; i < agent_chat_ids.size(); ++i)
      chat_int64[i].assign(agent_chat_ids[i].begin(), agent_chat_ids[i].end());

    auto probs_2d = lc_model->action_probs_from_vectors_std(obs_int64, chat_int64);
    probs_L.assign(N * 5, 0.0f);
    for (int i = 0; i < N; ++i)
      for (int a = 0; a < 5; ++a)
        probs_L[i * 5 + a] = probs_2d[static_cast<size_t>(i)][static_cast<size_t>(a)];
    softmax_per_agent(probs_L, N);

    auto dur_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - t_start).count();
    Common::STATS.policy_forward_calls += 1;
    Common::STATS.policy_forward_ms += dur_ms;
  }

  // Pick base distribution for FREE agents and global cache.
  // Priority order:
  //   LC-MAPF if we ran it (best signal for uncertain agents),
  //   else MAGAT (always available when magat_ok),
  //   else nothing → naive fallback.
  std::vector<float> free_dist;
  if (!probs_L.empty()) {
    free_dist = probs_L;
  } else if (magat_ok) {
    free_dist = probs_M;
  } else {
    // Neither MAGAT nor LC-MAPF available — degrade to naive.
    set_preferences_naive(Q);
    return;
  }

  // Build pin/free decision per agent.
  std::vector<int> pinned_action(N, -1);
  int n_pinned = 0;
  if (magat_ok) {
    for (int i = 0; i < N; ++i) {
      if (conf_M[i] >= MAGAT_THRESHOLD) {
        pinned_action[i] = magat_top[i];
        ++n_pinned;
      }
    }
  }
  // (If MAGAT didn't run, nothing gets pinned; everything is free under LC-MAPF.)

  // List of free agents (for variant rotation by scout).
  std::vector<int> free_agents;
  free_agents.reserve(N - n_pinned);
  for (int i = 0; i < N; ++i) {
    if (pinned_action[i] < 0) free_agents.push_back(i);
  }

  // Sorted-by-free_dist action lists for free-agent priorities (used by
  // variant rotation in build_preferences_from_ensemble_variant).
  std::vector<std::vector<int>> lc_mapf_sorted(N, std::vector<int>(5));
  for (int i = 0; i < N; ++i) {
    std::iota(lc_mapf_sorted[i].begin(), lc_mapf_sorted[i].end(), 0);
    std::sort(lc_mapf_sorted[i].begin(), lc_mapf_sorted[i].end(),
              [&](int a, int b) {
                return free_dist[i * 5 + a] > free_dist[i * 5 + b];
              });
  }

  // Uncertainty: 1 - n_pinned/N. High = many free agents = MAGAT uncertain =
  // scout should restart here for variant exploration.
  const float uncertainty = 1.0f - static_cast<float>(n_pinned) / static_cast<float>(N);

  // Cache the entry (only when stateless). Use unique_ptr for stable address
  // (entry_mtx, raw pointer held by scout loops).
  EnsembleCacheEntry *entry_ptr = nullptr;
  if (action_history == nullptr) {
    auto entry = std::make_unique<EnsembleCacheEntry>();
    entry->pinned_action = std::move(pinned_action);
    entry->lc_mapf_probs = free_dist;
    entry->lc_mapf_sorted = std::move(lc_mapf_sorted);
    entry->free_agents = std::move(free_agents);
    entry->n_pinned = n_pinned;
    entry->next_variant_idx = 0;
    entry->uncertainty = uncertainty;
    entry_ptr = entry.get();
    {
      std::lock_guard<std::mutex> lock(ensemble_cache_mtx);
      ensemble_cache[Q] = std::move(entry);
    }
  }

  // Build preferences for THIS call. If we just inserted entry, this consumes
  // variant 0 (= pinned + free-agent top actions); next caller gets variant 1.
  if (entry_ptr != nullptr) {
    build_preferences_from_ensemble_variant(Q, *entry_ptr);
    return;
  }

  // Stateless-with-history fallback: top variant only, no cache. Honors the
  // same pin/free split (pinned strongly biased to MAGAT top action).
  auto get_cost = [&](const int i, Vertex* v_from, Vertex* v_to) {
    const int v_idx = get_policy_action_index_from_vertex(v_from, v_to);
    float v_val = -free_dist[i * 5 + v_idx];
    const int pinned = pinned_action[i];
    if (pinned >= 0 && v_idx == pinned) v_val -= 2.0f;  // strong boost on MAGAT pin
    if (SAMPLING_STRATEGY == SamplingStrategy::Probablistic) {
      v_val = v_val / SAMPLING_TEMPERTURE - rng.get();
    }
    auto gg = global_guide.get(i, v_from, v_to);
    if (SAMPLING_STRATEGY == SamplingStrategy::Tiebreaking) {
      return std::make_pair(v_to,
                            std::make_tuple(gg, (float)D->get(i, v_to), v_val));
    } else {
      return std::make_pair(v_to, std::make_tuple(gg, v_val, rrd(MT)));
    }
  };
  // Re-derive pinned_action since we moved it into entry above. In this branch
  // (action_history != nullptr) entry_ptr is null, so the move did NOT happen.
  for (int i = 0; i < N; ++i) {
    const auto u = Q[i];
    const auto K = u->neighbor.size();
    for (size_t k = 0; k <= K; ++k) {
      preferences[i][k] = get_cost(i, u, u->actions[k]);
    }
    std::sort(preferences[i].begin(), preferences[i].begin() + K + 1,
              [&](auto&& a, auto&& b) { return std::get<1>(a) < std::get<1>(b); });
  }
}

EnsembleCacheEntry* AgentPolicy::get_ensemble_cache(const Config& Q)
{
  std::lock_guard<std::mutex> lock(ensemble_cache_mtx);
  auto it = ensemble_cache.find(Q);
  if (it == ensemble_cache.end()) return nullptr;
  return it->second.get();
}

Vertex* AgentPolicy::get(const int i, const int k)
{
  return std::get<0>(preferences[i][k]);
}

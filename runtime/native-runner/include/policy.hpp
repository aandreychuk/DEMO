#pragma once
#include <array>
#include <torch/script.h>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "dist_table.hpp"
#include "global_guide.hpp"
#include "graph.hpp"
#include "instance.hpp"
#include "rng.hpp"
#include "utils.hpp"

// Must match ACTION_HISTORY_LEN in lacam.hpp (last 5 POGEMA actions per agent)
constexpr int LC_MAPF_ACTION_HISTORY_LEN = 5;

using ActionCost = std::tuple<int, float, float>;
using Preference = std::vector<std::pair<Vertex *, ActionCost>>;
using Preferences = std::vector<Preference>;

constexpr int NO_AGENT = -1;

// Python/PyG HMAGAT bridge. Positions are packed as [x0, y0, x1, y1, ...]
// in map coordinates; the callback writes N*5 action scores in POGEMA action
// order (wait, north, south, west, east). A non-zero return code is an error.
using HMAGATInferenceCallback =
    int (*)(const int *positions_xy, int num_agents, float *action_scores,
            int num_actions);

extern "C" int set_hmagat_inference_callback(void *callback);
extern "C" int clear_hmagat_inference_callback();
bool hmagat_inference_available();
int run_hmagat_inference(const int *positions_xy, int num_agents,
                         float *action_scores, int num_actions);

// Forward declarations for LC-MAPF C++ core
class ObservationGenerator;
namespace lc_mapf {
class LCMAPFTorchscriptModel;
}  // namespace lc_mapf

// Forward declarations for MAPF-GPT-DDG C++ core
namespace mapf_gpt {
class ObservationGenerator;
class MAPFGPTTorchscriptModel;
}  // namespace mapf_gpt

// Forward declarations for DMM C++ core
// DMM reuses the LC-MAPF ObservationGenerator (identical observation format),
// so only the model wrapper is namespaced here.
namespace dmm {
class DMMTorchscriptModel;
}  // namespace dmm

// Forward declaration to break circular dep.
struct AgentPolicy;

// Per-thread policy configuration — replaces old static variables
struct PolicyConfig {
  std::string policy_type = "magat";       // "magat", "hmagat", "lc_mapf", "mapf_gpt", "dmm", or "ensemble"
  std::string model_filepath = "";         // MAGAT+ TorchScript model path
  std::string lc_mapf_model_path = "";     // LC-MAPF TorchScript model path
  std::string lc_mapf_device = "cuda";     // "cuda" or "cpu"
  std::string mapf_gpt_model_path = "";    // MAPF-GPT-DDG TorchScript model path
  std::string mapf_gpt_device = "cuda";    // "cuda" or "cpu"
  std::string dmm_model_path = "";         // DMMv2 TorchScript/AOTI model path
  std::string dmm_device = "cuda";         // "cuda" or "cpu"
  torch::jit::script::Module *model_ptr = nullptr;  // pre-loaded MAGAT+ model (for cloning)
  // Ensemble: list of sub-policy types to combine when policy_type == "ensemble".
  // Subset of {"magat","lc_mapf","mapf_gpt"}. Each named sub-policy uses its
  // respective model_path field above.
  std::vector<std::string> ensemble_members;
  // Pointer to the ensemble loop's AgentPolicy. Set by planner.cpp for
  // non-ensemble (scout) loops so they can read the ensemble cache for
  // variant-generation at restart targets. nullptr for the ensemble loop
  // itself and for legacy single-policy configs.
  AgentPolicy *ensemble_policy_ref = nullptr;
};

// Per-Config ensemble forward result cached for scout reuse.
//
// Strategy: MAGAT-first CASCADE (MAGAT confident → pin; otherwise consult
// LC-MAPF). Replaces older parallel consensus-pin (which always ran all 3
// NNs and pinned only on unanimous agreement). MAPF-GPT is intentionally
// NOT called in this 2-tier cascade (still loadable for future extension).
//
//   1. Forward MAGAT (~1.5 ms).
//   2. Per agent i, conf_M[i] = max_a P_MAGAT[i][a]. n_uncertain = #(conf < T).
//   3. If n_uncertain > 0: forward LC-MAPF (~5 ms). Else: skip.
//   4. Per agent: conf_M[i] >= T → pin agent i on argmax MAGAT.
//                conf_M[i] <  T → FREE; priorities = LC-MAPF (or MAGAT fallback).
//   T = MAGAT_THRESHOLD = 0.6 (hardcoded in policy.cpp).
//
// Perf rationale: when MAGAT is confident across all agents, we pay only
// MAGAT's ~1.5 ms instead of MAGAT+LC-MAPF+MAPF-GPT (~7.7 ms). Throughput
// gain motivated by maze-32-32-4 smoke results where always-3-NN ensemble
// lost ~16% SoC to LC-MAPF-only due to NN cost dominating wall time.
//
// Fields:
//   pinned_action[i]: 0..4 = forced action index for agent i (= MAGAT argmax
//     for confident agents), -1 = free.
//   lc_mapf_probs[N*5]: distribution used for FREE agents — LC-MAPF if it
//     ran, else MAGAT fallback. (Name retained for ABI / scout-loop compat.)
//   n_pinned: count of MAGAT-confident agents.
//   uncertainty = 1 - n_pinned/N. High = many free agents = MAGAT uncertain
//     = scout should restart here for more exploration.
//
// Variant state (next_variant_idx + per-call free-agent rotation):
//   Variant 0 = pinned MAGAT action for confident agents + top free_dist
//     action for everyone else.
//   Variant K = rotate one FREE agent's rank to next-best free_dist action.
//   Pinned agents NEVER change across variants — MAGAT was highly confident.
//
// entry_mtx guards next_variant_idx for concurrent scout calls.
struct EnsembleCacheEntry {
  std::vector<int> pinned_action;              // size N; -1 = free, 0..4 = forced action
  std::vector<float> lc_mapf_probs;            // size N*5; LC-MAPF distribution for free agents
  std::vector<std::vector<int>> lc_mapf_sorted;// [N][5] free-agent action indices desc by LC-MAPF prob
  std::vector<int> free_agents;                // list of free agent indices (for variant rotation)
  int n_pinned = 0;
  int next_variant_idx = 0;
  float uncertainty = 0.0f;
  mutable std::mutex entry_mtx;
};

struct AgentPolicy {
  enum SamplingStrategy {
    Deterministic,
    Probablistic,
    Tiebreaking,
  };

  const Instance *ins;
  std::mt19937 MT;
  std::uniform_real_distribution<float> rrd;  // random, real distribution

  // solver utils
  const int N;  // number of agents
  const int V_size;
  DistTable *D;
  std::vector<int> occupied_now;  // for quick location check
  bool use_model;
  std::string policy_type_;  // instance copy from PolicyConfig

  // RNG
  RandomNumberGenerator rng = RandomNumberGenerator();

  // inference
  const int fov_size;
  std::vector<torch::jit::IValue> inputs;
  torch::Device device;
  std::unordered_map<Config, torch::Tensor, ConfigHasher> known_config_table;
  torch::Tensor node_feature_;  // per-instance feature buffer

  // guidance
  GlobalGuide global_guide;

  // main
  Preferences preferences;

  // Per-instance MAGAT+ model (for thread-safe parallel execution)
  std::unique_ptr<torch::jit::script::Module> instance_model;

  // HMAGAT is evaluated by the official Python/PyG implementation through a
  // ctypes callback. LaGAT supplies the queried joint configuration and gets
  // one five-action score vector per agent back. The cache is essential:
  // LaCAM can revisit configurations, while HMAGAT hypergraph construction is
  // substantially more expensive than a C++ preference lookup.
  std::unordered_map<Config, std::vector<float>, ConfigHasher>
      hmagat_action_scores_cache;

  // hyper parameters (set once before threads start, safe to keep static)
  static int OBSERVATION_RAD;
  static bool USE_COMMUNICATION_RADIUS;
  static int COMMUNICATION_RADIUS;
  static SamplingStrategy SAMPLING_STRATEGY;
  static float SAMPLING_TEMPERTURE;

  // NN gate: if <= GATE_MAX_AMBIGUOUS agents have tied best/second-best
  // naive-distance preferences, skip the NN forward pass and use the
  // naive preferences directly. -1 disables gating (always call NN when
  // use_model is true). Default -1 (off); tune via --policy_nn_gate_max.
  static int NN_GATE_MAX_AMBIGUOUS;

  // Per-instance LC-MAPF state
  std::unique_ptr<ObservationGenerator> lc_obs_gen;
  std::unique_ptr<lc_mapf::LCMAPFTorchscriptModel> lc_model;
  bool lc_initialized = false;
  std::unordered_map<Config, std::vector<float>, ConfigHasher> lc_mapf_action_logits_cache;

  // Per-instance MAPF-GPT-DDG state
  std::unique_ptr<mapf_gpt::ObservationGenerator> mg_obs_gen;
  std::unique_ptr<mapf_gpt::MAPFGPTTorchscriptModel> mg_model;
  bool mg_initialized = false;
  std::unordered_map<Config, std::vector<float>, ConfigHasher> mapf_gpt_action_logits_cache;

  // Per-instance DMM state — note: dmm shares the LC-MAPF ObservationGenerator
  // (lc_obs_gen above), since the observation format is identical.
  std::unique_ptr<dmm::DMMTorchscriptModel> dmm_model;
  bool dmm_initialized = false;

  // Ensemble: which sub-policies are members + combined per-Config cache.
  // Populated only when policy_type_ == "ensemble".
  std::vector<std::string> ensemble_members_;
  // unique_ptr so EnsembleCacheEntry (which owns a mutex) has stable address
  // across map rehashes — scout loops hold pointers to entries.
  std::unordered_map<Config, std::unique_ptr<EnsembleCacheEntry>, ConfigHasher> ensemble_cache;
  std::mutex ensemble_cache_mtx;  // guards map structure during insert

  AgentPolicy(const Instance *_ins, DistTable *_D, const PolicyConfig &config, int seed = 0);
  ~AgentPolicy();

  void set_preferences(const Config &Q_from, const std::set<int> &A = {},
                      const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>> *action_history = nullptr);
  void set_preferences_naive(const Config &Q_from, const std::set<int> &A = {});
  void set_preferences_learned(const Config &Q_from, const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>> *action_history = nullptr);
  void set_preferences_learned_hmagat(const Config &Q_from);
  void set_preferences_learned_lc_mapf(const Config &Q_from, const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>> *action_history);
  void set_preferences_learned_mapf_gpt(const Config &Q_from, const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>> *action_history);
  void set_preferences_learned_dmm(const Config &Q_from, const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>> *action_history);
  // Run the DMM observation/model path without converting the five action
  // scores into LaCAM preferences. Used by the standalone policy rollouts.
  std::vector<std::vector<float>> get_dmm_action_probabilities(
      const Config &Q_from,
      const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>>
          *action_history = nullptr);
  void get_dmm_encoded_inputs(
      std::vector<std::vector<int>>& observations,
      std::vector<std::vector<int>>& chat) const;
  // Ensemble: MAGAT-first cascade. Run MAGAT; if any agent's MAGAT confidence
  // < MAGAT_THRESHOLD also run LC-MAPF; cache as EnsembleCacheEntry; build
  // preferences from pinned MAGAT actions + free-agent LC-MAPF priorities.
  void set_preferences_learned_ensemble(const Config &Q_from, const std::vector<std::array<int, LC_MAPF_ACTION_HISTORY_LEN>> *action_history);
  // Build preferences[] from cached entry's combined distribution + current
  // variant ranks. Caller is scout (or any) loop reusing ensemble's NN work.
  // Advances entry.next_variant_idx + ranks (under entry.entry_mtx) so the
  // next call produces the next variant.
  void build_preferences_from_ensemble_variant(const Config &Q, EnsembleCacheEntry &entry);

  // Public accessor: returns nullptr if Q not cached. Used by scout-loop to
  // detect "ensemble-visited" HNodes and read combined distribution.
  // Non-const because variant-aware callers acquire entry_mtx.
  EnsembleCacheEntry *get_ensemble_cache(const Config &Q);

  // Scout-loop ensemble pointer (mirror of PolicyConfig::ensemble_policy_ref).
  // When non-null and current Config Q has a cached entry, set_preferences
  // builds preferences from cached distribution with variant-bumped ranks
  // instead of running naive/own NN forward.
  AgentPolicy *ensemble_policy_ref_ = nullptr;

  // for model inference
  void set_features(const Config &Q);

  ActionCost get_action_cost(const int i, const Vertex *u);
  Vertex *get(const int i, const int k);
};

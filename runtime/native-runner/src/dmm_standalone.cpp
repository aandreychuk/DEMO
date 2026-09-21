// Standalone DMM model wrapper.
//
// DMM consumes the SAME observation format as LC-MAPF — token IDs from the
// Encoder in lc_mapf_standalone.cpp ([N, 256]) plus the chat_ids tensor from
// the same ObservationGenerator ([N, max_num_neighbors]). So this file only
// owns the TorchScript / AOTI model wrapper; LAGAT's policy.cpp shares the
// global LC-MAPF ObservationGenerator (lc_obs_gen) for both NN paths.
//
// Public surface (all inside `namespace dmm`):
//   class DMMTorchscriptModel
//     - ctor(model_path, device)            // .pt2 (AOTI dynamic), dir of
//                                           //   n<N>_bf16.pt2 (AOTI fixed-N
//                                           //   dispatch), or .pt (TorchScript)
//     - action_probs(obs, chat_ids)         // adds seeded stochastic inputs,
//                                             returns [N, 5]
//     - action_probs_from_vectors_std(...)  // public entry from policy.cpp
//     - greedy_actions_from_vectors(...)    // optional convenience
//
#include <torch/script.h>
#include <torch/version.h>
#if __has_include(<torch/csrc/inductor/aoti_package/model_package_loader.h>)
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#define LAGAT_DMM_HAS_AOTI_LOADER 1
#else
#define LAGAT_DMM_HAS_AOTI_LOADER 0
#endif
#if defined(LAGAT_DMM_CUDA_CONDITIONAL_GRAPH)
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "dmm_conditional_graph.h"
#endif
#include <filesystem>
#include <fstream>
#include <regex>
#include <vector>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <algorithm>
#include <stdexcept>
#include <cstdint>

namespace dmm {

class DMMTorchscriptModel {
public:
    static constexpr int64_t NUM_ACTIONS = 5;
    static constexpr int64_t NUM_COMM_ROUNDS = 4;
    static constexpr int64_t ADAPTIVE_MAX_ROUNDS = 16;

    DMMTorchscriptModel(const std::string& model_path,
                        const std::string& device = "cpu");
    ~DMMTorchscriptModel();

    // observations: [B, N, T], agent_chat_ids: [B, N, L] -> returns [N, 5]
    torch::Tensor action_probs(const torch::Tensor& observations,
                                const torch::Tensor& agent_chat_ids,
                                const torch::Tensor& active_mask = {});

    // observations [N][T], chat_ids [N][L] -> [N][5]
    std::vector<std::vector<float>> action_probs_from_vectors_std(
        const std::vector<std::vector<int64_t>>& observations,
        const std::vector<std::vector<int64_t>>& agent_chat_ids,
        const std::vector<char>& active_mask = {});

    // Argmax action per agent.
    std::vector<int64_t> greedy_actions_from_vectors(
        const std::vector<std::vector<int64_t>>& observations,
        const std::vector<std::vector<int64_t>>& agent_chat_ids);

private:
    torch::jit::script::Module module_;
    torch::Device device_;

    // AOTI mode: a fixed agents_<N>.pt2 package, a directory of fixed
    // packages, or a legacy dynamic/four-input package.
    bool aoti_mode_ = false;
    bool adaptive_aoti_mode_ = false;
    bool deterministic_two_input_aoti_ = false;
#if LAGAT_DMM_HAS_AOTI_LOADER
    std::vector<std::pair<int, std::unique_ptr<torch::inductor::AOTIModelPackageLoader>>> aoti_loaders_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> adaptive_init_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader> adaptive_round_;
    std::unique_ptr<torch::inductor::AOTIModelPackageLoader>
        adaptive_device_step_;
#endif
    long long adaptive_calls_ = 0;
    long long adaptive_rounds_total_ = 0;
    int64_t adaptive_rounds_max_ = 0;
#if defined(LAGAT_DMM_CUDA_CONDITIONAL_GRAPH)
    int64_t adaptive_graph_num_agents_ = -1;
    torch::Tensor graph_obs_;
    torch::Tensor graph_initial_exponential_;
    torch::Tensor graph_latent_;
    torch::Tensor graph_z_;
    torch::Tensor graph_h_;
    torch::Tensor graph_chat_;
    torch::Tensor graph_vote_gumbels_;
    torch::Tensor graph_active_;
    torch::Tensor graph_previous_probs_;
    torch::Tensor graph_previous_actions_;
    torch::Tensor graph_consecutive_stable_;
    torch::Tensor graph_round_index_;
    std::vector<torch::Tensor> graph_step_outputs_;
    std::vector<torch::Tensor> graph_init_outputs_;
    std::unique_ptr<at::cuda::CUDAGraph> adaptive_init_graph_;
    std::unique_ptr<at::cuda::CUDAGraph> adaptive_body_graph_;
    DMMConditionalGraph* adaptive_conditional_graph_ = nullptr;
#endif

    static torch::Tensor vector2d_to_long_tensor(
        const std::vector<std::vector<int64_t>>& data);
    static torch::Device parse_device(const std::string& device_name);
    static void check_batched_shapes(const torch::Tensor& observations,
                                      const torch::Tensor& agent_chat_ids);
};

DMMTorchscriptModel::DMMTorchscriptModel(const std::string& model_path,
                                          const std::string& device)
    : device_(parse_device(device)) {
    namespace fs = std::filesystem;
    if (fs::exists(model_path) && fs::is_regular_file(model_path)
        && model_path.size() >= 4
        && model_path.compare(model_path.size() - 4, 4, ".pt2") == 0) {
#if LAGAT_DMM_HAS_AOTI_LOADER
        if (!device_.is_cuda()) {
            throw std::runtime_error(
                "DMM AOTI .pt2 mode requires CUDA, got: " + device);
        }
        std::smatch match;
        const std::string filename = fs::path(model_path).filename().string();
        const std::regex deterministic_pt2_re(R"(agents_(\d+)\.pt2)");
        const bool fixed_deterministic_package =
            std::regex_match(filename, match, deterministic_pt2_re);
        bool manifest_two_input = false;
        std::ifstream manifest_stream(model_path + ".json");
        if (manifest_stream) {
            const std::string manifest(
                (std::istreambuf_iterator<char>(manifest_stream)),
                std::istreambuf_iterator<char>());
            manifest_two_input =
                manifest.find("\"input_contract\": \"obs-chat-v1\"") !=
                std::string::npos;
        }
        const bool deterministic_package =
            fixed_deterministic_package || manifest_two_input;
        const int fixed_num_agents =
            fixed_deterministic_package ? std::stoi(match[1].str()) : 0;
        std::cerr << "[dmm] AOTI loading " << model_path << std::endl;
        aoti_loaders_.emplace_back(fixed_num_agents,
            std::make_unique<torch::inductor::AOTIModelPackageLoader>(model_path));
        deterministic_two_input_aoti_ = deterministic_package;
        aoti_mode_ = true;
        std::cerr << "[dmm] AOTI "
                  << (deterministic_package
                          ? (fixed_deterministic_package
                                 ? "fixed-N obs-chat-v1"
                                 : "dynamic-N obs-chat-v1")
                          : "dynamic-N explicit-random-v1")
                  << " mode active" << std::endl;
        return;
#else
        throw std::runtime_error(
            "This libtorch does not provide the C++ AOTI package loader; "
            "use a TorchScript .pt model or upgrade libtorch");
#endif
    }
    if (fs::exists(model_path) && fs::is_directory(model_path)) {
#if LAGAT_DMM_HAS_AOTI_LOADER
        if (!device_.is_cuda()) {
            throw std::runtime_error(
                "DMM AOTI mode requires CUDA, got: " + device);
        }
        const auto adaptive_init_path =
            fs::path(model_path) / "adaptive_init.pt2";
        const auto adaptive_round_path =
            fs::path(model_path) / "adaptive_round.pt2";
        if (fs::is_regular_file(adaptive_init_path) &&
            fs::is_regular_file(adaptive_round_path)) {
            std::cerr << "[dmm] AOTI adaptive loading "
                      << adaptive_init_path << " and "
                      << adaptive_round_path << std::endl;
            adaptive_init_ =
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
                std::make_unique<torch::inductor::AOTIModelPackageLoader>(
                    adaptive_init_path.string(),
                    "model",
                    /*run_single_threaded=*/true,
                    /*num_runners=*/1,
                    device_.index());
#else
                std::make_unique<torch::inductor::AOTIModelPackageLoader>(
                    adaptive_init_path.string());
#endif
            adaptive_round_ =
                std::make_unique<torch::inductor::AOTIModelPackageLoader>(
                    adaptive_round_path.string());
#if defined(LAGAT_DMM_CUDA_CONDITIONAL_GRAPH)
            const auto adaptive_device_step_path =
                fs::path(model_path) / "adaptive_device_step.pt2";
            if (fs::is_regular_file(adaptive_device_step_path)) {
                adaptive_device_step_ =
#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 7)
                    std::make_unique<
                        torch::inductor::AOTIModelPackageLoader>(
                        adaptive_device_step_path.string(),
                        "model",
                        /*run_single_threaded=*/true,
                        /*num_runners=*/1,
                        device_.index());
#else
                    std::make_unique<
                        torch::inductor::AOTIModelPackageLoader>(
                        adaptive_device_step_path.string());
#endif
                std::cerr
                    << "[dmm] GPU-resident adaptive conditional graph enabled: "
                    << adaptive_device_step_path << std::endl;
            }
#endif
            adaptive_aoti_mode_ = true;
            aoti_mode_ = true;
            return;
        }
        const std::regex legacy_pt2_re(R"(n(\d+)_bf16\.pt2)");
        const std::regex deterministic_pt2_re(R"(agents_(\d+)\.pt2)");
        bool saw_legacy_package = false;
        bool saw_deterministic_package = false;
        for (const auto& entry : fs::directory_iterator(model_path)) {
            std::smatch m;
            std::string fname = entry.path().filename().string();
            bool deterministic_package =
                std::regex_match(fname, m, deterministic_pt2_re);
            if (!deterministic_package &&
                !std::regex_match(fname, m, legacy_pt2_re)) {
                continue;
            }
            if (deterministic_package) {
                saw_deterministic_package = true;
            } else {
                saw_legacy_package = true;
            }
            if (saw_deterministic_package && saw_legacy_package) {
                throw std::runtime_error(
                    "DMM AOTI directory mixes two-input agents_<N>.pt2 "
                    "and four-input n<N>_bf16.pt2 packages");
            }
            {
                int n = std::stoi(m[1].str());
                std::cerr << "[dmm] AOTI: loading n=" << n << " from " << entry.path() << std::endl;
                aoti_loaders_.emplace_back(n,
                    std::make_unique<torch::inductor::AOTIModelPackageLoader>(entry.path().string()));
            }
        }
        if (aoti_loaders_.empty()) {
            throw std::runtime_error(
                "DMM AOTI: no n<N>_bf16.pt2 found in " + model_path);
        }
        std::sort(aoti_loaders_.begin(), aoti_loaders_.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        deterministic_two_input_aoti_ = saw_deterministic_package;
        aoti_mode_ = true;
        std::cerr << "[dmm] AOTI mode active with " << aoti_loaders_.size()
                  << " packages, max N=" << aoti_loaders_.back().first
                  << ", contract="
                  << (deterministic_two_input_aoti_ ? "obs-chat-v1"
                                                    : "explicit-random-v1")
                  << std::endl;
        return;
#else
        throw std::runtime_error(
            "This libtorch does not provide the C++ AOTI package loader; "
            "use a TorchScript .pt model or upgrade libtorch");
#endif
    }

    module_ = torch::jit::load(model_path);
    module_.to(device_);
    module_.eval();
}

DMMTorchscriptModel::~DMMTorchscriptModel() {
#if defined(LAGAT_DMM_CUDA_CONDITIONAL_GRAPH)
    dmm_conditional_graph_destroy(adaptive_conditional_graph_);
    adaptive_conditional_graph_ = nullptr;
#endif
    if (!adaptive_aoti_mode_ || adaptive_calls_ == 0) return;
    std::cerr << "[dmm-adaptive-summary] calls=" << adaptive_calls_
              << " rounds_total=" << adaptive_rounds_total_
              << " rounds_mean="
              << static_cast<double>(adaptive_rounds_total_) /
                     static_cast<double>(adaptive_calls_)
              << " rounds_max=" << adaptive_rounds_max_ << std::endl;
}

torch::Device DMMTorchscriptModel::parse_device(const std::string& device_name) {
    if (device_name == "cpu")
        return torch::Device(torch::kCPU);
    if (device_name.rfind("cuda", 0) == 0)
        return torch::Device(device_name);
    if (device_name == "mps")
        return torch::Device(torch::kMPS);
    throw std::invalid_argument("Unsupported device string: " + device_name);
}

void DMMTorchscriptModel::check_batched_shapes(
    const torch::Tensor& observations,
    const torch::Tensor& agent_chat_ids) {
    if (observations.dim() != 3) {
        throw std::invalid_argument("observations must be rank-3 [B, N, T]");
    }
    if (agent_chat_ids.dim() != 3) {
        throw std::invalid_argument("agent_chat_ids must be rank-3 [B, N, L]");
    }
    if (observations.size(0) != agent_chat_ids.size(0) ||
        observations.size(1) != agent_chat_ids.size(1)) {
        throw std::invalid_argument(
            "observations and agent_chat_ids must have matching [B, N]");
    }
}

torch::Tensor DMMTorchscriptModel::vector2d_to_long_tensor(
    const std::vector<std::vector<int64_t>>& data) {
    if (data.empty())
        throw std::invalid_argument("Input 2D vector must not be empty");
    const int64_t rows = static_cast<int64_t>(data.size());
    const int64_t cols = static_cast<int64_t>(data.front().size());
    if (cols == 0)
        throw std::invalid_argument("Input 2D vector must have non-empty rows");

    std::vector<int64_t> flat;
    flat.reserve(static_cast<size_t>(rows * cols));
    for (const auto& row : data) {
        if (static_cast<int64_t>(row.size()) != cols)
            throw std::invalid_argument("Input 2D vector rows must all have equal length");
        flat.insert(flat.end(), row.begin(), row.end());
    }
    return torch::from_blob(flat.data(), {rows, cols},
                            torch::TensorOptions().dtype(torch::kLong)).clone();
}

torch::Tensor DMMTorchscriptModel::action_probs(
    const torch::Tensor& observations,
    const torch::Tensor& agent_chat_ids,
    const torch::Tensor& active_mask) {
    c10::InferenceMode guard(true);
    check_batched_shapes(observations, agent_chat_ids);

    auto obs = observations.to(device_, torch::kLong, /*non_blocking=*/false, /*copy=*/false);
    auto chat = agent_chat_ids.to(device_, torch::kLong, /*non_blocking=*/false, /*copy=*/false);
    const int64_t batch_size = obs.size(0);
    const int64_t num_agents = obs.size(1);
    const auto random_options =
        torch::TensorOptions().dtype(torch::kFloat32).device(device_);

    // Dirichlet(1) is normalized iid Exp(1).  Gumbel-max samples exactly
    // from Categorical(logits / tau).  Randomness stays outside the exported
    // graph so AOTI has a stable, portable four-input contract.
    torch::Tensor initial_exponential;
    torch::Tensor vote_gumbels;
    if (!deterministic_two_input_aoti_) {
        initial_exponential =
            torch::empty({batch_size, num_agents, NUM_ACTIONS}, random_options)
                .exponential_(1.0);
        vote_gumbels =
            torch::empty(
                {adaptive_aoti_mode_ ? ADAPTIVE_MAX_ROUNDS : NUM_COMM_ROUNDS,
                 batch_size, num_agents, NUM_ACTIONS},
                random_options)
                .exponential_(1.0)
                .log()
                .neg();
    }

    if (aoti_mode_) {
#if LAGAT_DMM_HAS_AOTI_LOADER
        if (adaptive_aoti_mode_) {
            bool need_init_outputs = true;
#if defined(LAGAT_DMM_CUDA_CONDITIONAL_GRAPH)
            const bool device_graph_rebuild =
                adaptive_device_step_ &&
                (adaptive_conditional_graph_ == nullptr ||
                 adaptive_graph_num_agents_ != num_agents);
            need_init_outputs =
                !adaptive_device_step_ || device_graph_rebuild;
#endif
            std::vector<torch::Tensor> init_outs;
            if (need_init_outputs) {
                init_outs =
                    adaptive_init_->run({obs, initial_exponential});
                if (init_outs.size() != 3) {
                    throw std::runtime_error(
                        "DMM adaptive init must return latent, z, and h");
                }
            }
            torch::Tensor latent =
                need_init_outputs ? init_outs[0] : torch::Tensor();
            torch::Tensor z =
                need_init_outputs ? init_outs[1] : torch::Tensor();
            torch::Tensor h =
                need_init_outputs ? init_outs[2] : torch::Tensor();
            auto active = active_mask.defined()
                ? active_mask.to(device_, torch::kBool).reshape({-1})
                : torch::ones(
                      {batch_size * num_agents},
                      torch::TensorOptions()
                          .dtype(torch::kBool)
                          .device(device_));

#if defined(LAGAT_DMM_CUDA_CONDITIONAL_GRAPH)
            if (adaptive_device_step_) {
                int graph_device_index = device_.index();
                if (graph_device_index < 0) {
                    cudaGetDevice(&graph_device_index);
                }
                if (device_graph_rebuild) {
                    dmm_conditional_graph_destroy(
                        adaptive_conditional_graph_);
                    adaptive_conditional_graph_ = nullptr;
                    adaptive_init_graph_.reset();
                    adaptive_body_graph_.reset();
                    graph_init_outputs_.clear();
                    graph_step_outputs_.clear();

                    adaptive_graph_num_agents_ = num_agents;
                    graph_obs_ = torch::empty_like(obs);
                    graph_initial_exponential_ =
                        torch::empty_like(initial_exponential);
                    graph_latent_ = torch::empty_like(latent);
                    graph_z_ = torch::empty_like(z);
                    graph_h_ = torch::empty_like(h);
                    graph_chat_ = torch::empty_like(chat);
                    graph_vote_gumbels_ =
                        torch::empty_like(vote_gumbels);
                    graph_active_ = torch::empty_like(active);
                    graph_previous_probs_ = torch::full(
                        {batch_size * num_agents, NUM_ACTIONS},
                        1.0f / static_cast<float>(NUM_ACTIONS),
                        random_options);
                    graph_previous_actions_ = torch::zeros(
                        {batch_size * num_agents},
                        torch::TensorOptions()
                            .dtype(torch::kLong)
                            .device(device_));
                    graph_consecutive_stable_ = torch::zeros(
                        {1},
                        torch::TensorOptions()
                            .dtype(torch::kLong)
                            .device(device_));
                    graph_round_index_ = torch::zeros_like(
                        graph_consecutive_stable_);

                    graph_obs_.copy_(obs);
                    graph_initial_exponential_.copy_(
                        initial_exponential);
                    graph_latent_.copy_(latent);
                    graph_z_.copy_(z);
                    graph_h_.copy_(h);
                    graph_chat_.copy_(chat);
                    graph_vote_gumbels_.copy_(vote_gumbels);
                    graph_active_.copy_(active);

                    // Warm the package before capture so lazy loader work and
                    // allocator setup are not accidentally included.
                    auto warmup_outputs = adaptive_device_step_->run(
                        {graph_latent_,
                         graph_z_,
                         graph_h_,
                         graph_chat_,
                         graph_vote_gumbels_,
                         graph_active_,
                         graph_previous_probs_,
                         graph_previous_actions_,
                         graph_consecutive_stable_,
                         graph_round_index_});
                    if (warmup_outputs.size() != 7) {
                        throw std::runtime_error(
                            "DMM adaptive device step must return 7 tensors");
                    }
                    auto current_stream =
                        c10::cuda::getCurrentCUDAStream(
                            graph_device_index);
                    cudaStreamSynchronize(current_stream.stream());

                    auto capture_stream =
                        c10::cuda::getStreamFromPool(
                            /*high_priority=*/false,
                            graph_device_index);
                    {
                        c10::cuda::CUDAStreamGuard stream_guard(
                            capture_stream);
                        adaptive_init_graph_ =
                            std::make_unique<at::cuda::CUDAGraph>(
                                /*keep_graph=*/true);
                        adaptive_init_graph_->capture_begin();
                        graph_init_outputs_ = adaptive_init_->run(
                            {graph_obs_, graph_initial_exponential_},
                            reinterpret_cast<void*>(
                                capture_stream.stream()));
                        graph_latent_.copy_(graph_init_outputs_[0]);
                        graph_z_.copy_(graph_init_outputs_[1]);
                        graph_h_.copy_(graph_init_outputs_[2]);
                        adaptive_init_graph_->capture_end();
                    }

                    adaptive_conditional_graph_ =
                        dmm_conditional_graph_begin(
                            adaptive_init_graph_->raw_cuda_graph(),
                            graph_round_index_.data_ptr<int64_t>(),
                            graph_consecutive_stable_.data_ptr<int64_t>(),
                            graph_device_index);
                    const auto conditional_handle =
                        dmm_conditional_graph_handle(
                            adaptive_conditional_graph_);

                    {
                        c10::cuda::CUDAStreamGuard stream_guard(
                            capture_stream);
                        adaptive_body_graph_ =
                            std::make_unique<at::cuda::CUDAGraph>(
                                /*keep_graph=*/true);
                        adaptive_body_graph_->capture_begin();
                        graph_step_outputs_ =
                            adaptive_device_step_->run(
                                {graph_latent_,
                                 graph_z_,
                                 graph_h_,
                                 graph_chat_,
                                 graph_vote_gumbels_,
                                 graph_active_,
                                 graph_previous_probs_,
                                 graph_previous_actions_,
                                 graph_consecutive_stable_,
                                 graph_round_index_},
                                reinterpret_cast<void*>(
                                    capture_stream.stream()));
                        graph_z_.copy_(graph_step_outputs_[0]);
                        graph_h_.copy_(graph_step_outputs_[1]);
                        graph_previous_probs_.copy_(
                            graph_step_outputs_[2]);
                        graph_previous_actions_.copy_(
                            graph_step_outputs_[3]);
                        graph_consecutive_stable_.copy_(
                            graph_step_outputs_[4]);
                        graph_round_index_.copy_(
                            graph_step_outputs_[5]);
                        dmm_conditional_graph_set_condition(
                            conditional_handle,
                            graph_step_outputs_[6].data_ptr<bool>(),
                            capture_stream.stream());
                        adaptive_body_graph_->capture_end();
                    }
                    dmm_conditional_graph_finalize(
                        adaptive_conditional_graph_,
                        adaptive_body_graph_->raw_cuda_graph());
                    std::cerr
                        << "[dmm] captured GPU-resident adaptive WHILE "
                        << "for N=" << num_agents << std::endl;
                }

                // Per-policy-call inputs are copied asynchronously.  The
                // parent graph resets round/patience state, then performs the
                // exact device-selected number of communication rounds.
                graph_obs_.copy_(obs);
                graph_initial_exponential_.copy_(
                    initial_exponential);
                graph_chat_.copy_(chat);
                graph_vote_gumbels_.copy_(vote_gumbels);
                graph_active_.copy_(active);
                auto stream = c10::cuda::getCurrentCUDAStream(
                    graph_device_index);
                dmm_conditional_graph_launch(
                    adaptive_conditional_graph_, stream.stream());
                return torch::softmax(
                    graph_z_.to(torch::kFloat32), -1);
            }
#endif

            torch::Tensor previous_probs;
            torch::Tensor previous_actions;
            int consecutive_stable = 0;
            int64_t rounds_used = 0;

            for (int64_t round_idx = 0;
                 round_idx < ADAPTIVE_MAX_ROUNDS; ++round_idx) {
                auto round_gumbel =
                    vote_gumbels.slice(0, round_idx, round_idx + 1);
                auto round_outs = adaptive_round_->run(
                    {latent, z, h, chat, round_gumbel});
                if (round_outs.size() != 3) {
                    throw std::runtime_error(
                        "DMM adaptive round must return logits, z, and h");
                }
                auto logits = round_outs[0];
                z = round_outs[1];
                h = round_outs[2];
                auto probs = torch::softmax(
                    logits.to(torch::kFloat32), -1);
                auto actions = z.argmax(-1);
                rounds_used = round_idx + 1;

                bool stable = false;
                if (previous_probs.defined()) {
                    if (!active.any().item<bool>()) {
                        stable = true;
                    } else {
                        constexpr float tiny = 1.17549435e-38f;
                        auto current = probs.clamp_min(tiny);
                        auto previous = previous_probs.clamp_min(tiny);
                        current = current /
                            current.sum(-1, true);
                        previous = previous /
                            previous.sum(-1, true);
                        auto midpoint = 0.5f * (current + previous);
                        auto js = 0.5f * (
                            (current * (current.log() - midpoint.log()))
                                .sum(-1) +
                            (previous * (previous.log() - midpoint.log()))
                                .sum(-1));
                        auto active_js = js.masked_select(active);
                        const float js_q =
                            torch::quantile(active_js, 0.95)
                                .item<float>();
                        const float action_fraction =
                            actions.eq(previous_actions)
                                .masked_select(active)
                                .to(torch::kFloat32)
                                .mean()
                                .item<float>();
                        stable =
                            js_q <= 0.05f && action_fraction >= 0.95f;
                    }
                    consecutive_stable =
                        stable ? consecutive_stable + 1 : 0;
                }
                previous_probs = probs;
                previous_actions = actions;

                if (rounds_used >= 2 && consecutive_stable >= 2) {
                    break;
                }
            }
            ++adaptive_calls_;
            adaptive_rounds_total_ += rounds_used;
            adaptive_rounds_max_ =
                std::max(adaptive_rounds_max_, rounds_used);
            return torch::softmax(z.to(torch::kFloat32), -1);
        }
        const int64_t actual_n = obs.size(1);

        // Dynamic-N single-package: no dispatch.
        if (aoti_loaders_.size() == 1 && aoti_loaders_.front().first == 0) {
            auto outs = deterministic_two_input_aoti_
                ? aoti_loaders_.front().second->run({obs, chat})
                : aoti_loaders_.front().second->run(
                      {obs, chat, initial_exponential, vote_gumbels});
            if (outs.empty())
                throw std::runtime_error("DMM AOTI dynamic: runner returned no outputs");
            return outs[0];
        }

        // Fixed-N dispatch mode.
        auto it = std::lower_bound(
            aoti_loaders_.begin(), aoti_loaders_.end(), actual_n,
            [](const auto& p, int64_t n) { return p.first < n; });
        if (it == aoti_loaders_.end()) {
            throw std::runtime_error(
                "DMM AOTI: no .pt2 for N=" + std::to_string(actual_n)
                + " (max compiled N=" + std::to_string(aoti_loaders_.back().first) + ")");
        }
        const int64_t target_n = it->first;
        torch::Tensor obs_in = obs;
        torch::Tensor chat_in = chat;
        if (actual_n < target_n) {
            const int64_t pad = target_n - actual_n;
            auto pad_obs = torch::full(
                {obs.size(0), pad, obs.size(2)}, 66,
                torch::TensorOptions().dtype(obs.dtype()).device(obs.device()));
            auto pad_chat = torch::full(
                {chat.size(0), pad, chat.size(2)}, -1,
                torch::TensorOptions().dtype(chat.dtype()).device(chat.device()));
            obs_in = torch::cat({obs, pad_obs}, /*dim=*/1);
            chat_in = torch::cat({chat, pad_chat}, /*dim=*/1);
            if (!deterministic_two_input_aoti_) {
                auto pad_initial = torch::ones(
                    {initial_exponential.size(0), pad,
                     initial_exponential.size(2)},
                    random_options);
                auto pad_gumbels = torch::zeros(
                    {vote_gumbels.size(0), vote_gumbels.size(1), pad,
                     vote_gumbels.size(3)},
                    random_options);
                initial_exponential =
                    torch::cat({initial_exponential, pad_initial}, /*dim=*/1);
                vote_gumbels =
                    torch::cat({vote_gumbels, pad_gumbels}, /*dim=*/2);
            }
        }
        auto outs = deterministic_two_input_aoti_
            ? it->second->run({obs_in, chat_in})
            : it->second->run(
                  {obs_in, chat_in, initial_exponential, vote_gumbels});
        if (outs.empty())
            throw std::runtime_error("DMM AOTI: runner returned no outputs");
        torch::Tensor result = outs[0];
        if (actual_n < target_n)
            result = result.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_n);
        return result;
#else
        throw std::runtime_error("DMM AOTI mode is unavailable in this libtorch");
#endif
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(obs);
    inputs.emplace_back(chat);
    inputs.emplace_back(initial_exponential);
    inputs.emplace_back(vote_gumbels);
    return module_.forward(inputs).toTensor();
}

std::vector<std::vector<float>> DMMTorchscriptModel::action_probs_from_vectors_std(
    const std::vector<std::vector<int64_t>>& observations,
    const std::vector<std::vector<int64_t>>& agent_chat_ids,
    const std::vector<char>& active_mask) {
    static const bool dmm_profile = std::getenv("LAGAT_DMM_PROFILE") != nullptr;
    static long long t_build_us = 0, t_forward_us = 0, t_d2h_us = 0, t_pack_us = 0;
    static int call_idx = 0;
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto us = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };

    auto t0 = now();
    auto obs = vector2d_to_long_tensor(observations).unsqueeze(0);   // [1, N, T]
    auto chat = vector2d_to_long_tensor(agent_chat_ids).unsqueeze(0); // [1, N, L]
    torch::Tensor active;
    if (!active_mask.empty()) {
        active = torch::from_blob(
                     const_cast<char*>(active_mask.data()),
                     {1, static_cast<int64_t>(active_mask.size())},
                     torch::TensorOptions().dtype(torch::kInt8))
                     .clone()
                     .to(torch::kBool);
    }
    auto t1 = now();
    auto probs_gpu = action_probs(obs, chat, active);
    auto t2 = now();
    // AOTI BF16 packages return BF16 scores.  Normalize the host-side ABI to
    // float before accessing data_ptr<float>().
    auto probs = probs_gpu.to(torch::kCPU, torch::kFloat32).contiguous();
    auto t3 = now();

#if defined(LAGAT_DMM_CUDA_CONDITIONAL_GRAPH)
    // The probability copy above has already synchronized this inference.
    // Reading the device loop counter here therefore adds no mid-loop or
    // additional GPU synchronization.
    if (adaptive_conditional_graph_ != nullptr &&
        graph_round_index_.defined()) {
        const int64_t rounds_used =
            graph_round_index_.to(torch::kCPU).item<int64_t>();
        ++adaptive_calls_;
        adaptive_rounds_total_ += rounds_used;
        adaptive_rounds_max_ =
            std::max(adaptive_rounds_max_, rounds_used);
    }
#endif

    if (probs.dim() != 2)
        throw std::runtime_error("Expected 2D probability tensor [N, 5]");
    const int64_t rows = probs.size(0);
    const int64_t cols = probs.size(1);
    const float* data = probs.data_ptr<float>();
    std::vector<std::vector<float>> out(static_cast<size_t>(rows),
                                         std::vector<float>(static_cast<size_t>(cols)));
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            out[static_cast<size_t>(i)][static_cast<size_t>(j)] = data[i * cols + j];
        }
    }
    auto t4 = now();

    if (dmm_profile) {
        t_build_us   += us(t0, t1);
        t_forward_us += us(t1, t2);
        t_d2h_us     += us(t2, t3);
        t_pack_us    += us(t3, t4);
        ++call_idx;
        if (call_idx % 50 == 0) {
            std::cerr << "[dmm-fwd-profile] calls=" << call_idx
                      << " sums(ms): vec_to_tensor=" << t_build_us / 1000
                      << " forward=" << t_forward_us / 1000
                      << " d2h=" << t_d2h_us / 1000
                      << " pack_vec=" << t_pack_us / 1000
                      << std::endl;
        }
    }
    return out;
}

std::vector<int64_t> DMMTorchscriptModel::greedy_actions_from_vectors(
    const std::vector<std::vector<int64_t>>& observations,
    const std::vector<std::vector<int64_t>>& agent_chat_ids) {
    auto obs = vector2d_to_long_tensor(observations).unsqueeze(0);
    auto chat = vector2d_to_long_tensor(agent_chat_ids).unsqueeze(0);
    auto probs = action_probs(obs, chat);
    auto actions = std::get<1>(probs.max(-1, false)).to(torch::kCPU).contiguous();
    return std::vector<int64_t>(actions.data_ptr<int64_t>(),
                                 actions.data_ptr<int64_t>() + actions.numel());
}

}  // namespace dmm

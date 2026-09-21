// Standalone C++ MAPF-GPT-DDG core: observation generator + TorchScript model.
// Observation format is identical to LC-MAPF but without communication module
// (no agent_chat_ids input to the model).
//
// All types are in the mapf_gpt namespace to avoid collisions with the
// LC-MAPF standalone included in the same translation unit (policy.cpp).

#include <torch/script.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <array>
#include <vector>
#include <queue>
#include <deque>
#include <iostream>
#include <string>
#include <unordered_map>
#include <filesystem>
#include <regex>
#include <memory>
#include <algorithm>
#include <bitset>
#include <algorithm>
#include <sstream>
#include <limits>
#include <cstdint>
#include <stdexcept>

namespace mapf_gpt {

struct InputParameters {
    InputParameters(int cvl = 20, int na = 13, int npa = 5, int cs = 256,
                    int obsr = 5, int ar = 5, int tt = -1)
        : cost2go_value_limit(cvl), num_agents(na), num_previous_actions(npa),
          context_size(cs), obs_radius(obsr), agents_radius(ar), task_type_id(tt) {}
    int cost2go_value_limit;
    int num_agents;
    int num_previous_actions;
    int context_size;
    int obs_radius;
    int agents_radius;
    int task_type_id;
};

struct AgentsInfo {
    AgentsInfo(std::pair<int,int> rp, std::pair<int,int> rg,
               std::deque<std::string> pa, std::string na)
        : relative_pos(rp), relative_goal(rg), previous_actions(pa), next_action(na) {}
    AgentsInfo() {}
    std::pair<int,int> relative_pos;
    std::pair<int,int> relative_goal;
    std::deque<std::string> previous_actions;
    std::string next_action;
};

struct Agent {
    std::pair<int,int> pos;
    std::pair<int,int> goal;
    std::deque<std::string> action_history;
    std::string next_action;
};

class Encoder {
public:
    InputParameters cfg;
    std::vector<int> coord_range;
    std::vector<char> actions_range;
    std::vector<std::string> next_action_range;
    std::unordered_map<std::string, int> str_vocab;
    std::unordered_map<int, int> int_vocab;

    Encoder(const InputParameters &cfg);
    std::vector<int> encode(const std::vector<AgentsInfo> &agents,
                            const std::vector<std::vector<int>> &cost2go);
};

class ObservationGenerator {
public:
    std::vector<Agent> agents;
    InputParameters cfg;
    Encoder encoder;
    std::vector<std::vector<int>> agents_locations;
    std::vector<std::vector<std::vector<int>>> cost2go_obs_buffer;
    std::vector<std::vector<int>> grid;
    std::vector<std::vector<std::vector<uint16_t>>> agent_cost2go;

    ObservationGenerator(const std::vector<std::vector<int>> &grid,
                         const InputParameters &cfg)
        : grid(grid), cfg(cfg), encoder(cfg)
    {
        agents_locations = std::vector<std::vector<int>>(
            grid.size(), std::vector<int>(grid[0].size(), -1));
    }
    ~ObservationGenerator() {}

    void compute_agent_cost2go(int agent_idx);
    void generate_cost2go_obs(int agent_idx, std::vector<std::vector<int>> &buffer);
    int get_distance(int agent_idx, const std::pair<int,int> &pos);
    void create_agents(const std::vector<std::pair<int,int>> &positions,
                       const std::vector<std::pair<int,int>> &goals);
    void update_next_action(int agent_idx);
    void update_agents(const std::vector<std::pair<int,int>> &positions,
                       const std::vector<std::pair<int,int>> &goals,
                       const std::vector<int> &actions);
    void update_agents(const std::vector<std::pair<int,int>> &positions,
                       const std::vector<std::pair<int,int>> &goals,
                       const std::vector<std::array<int,5>> &action_histories);
    std::vector<AgentsInfo> get_agents_info(int agent_idx);
    std::vector<std::vector<int>> generate_observations();
};

// ---------------------------------------------------------------------------
// Encoder implementation
// ---------------------------------------------------------------------------

Encoder::Encoder(const InputParameters &cfg) : cfg(cfg) {
    for (int i = -cfg.cost2go_value_limit; i <= cfg.cost2go_value_limit; ++i)
        coord_range.push_back(i);
    coord_range.push_back(-cfg.cost2go_value_limit * 4);
    coord_range.push_back(-cfg.cost2go_value_limit * 2);
    coord_range.push_back(cfg.cost2go_value_limit * 2);

    actions_range = {'n', 'w', 'u', 'd', 'l', 'r'};
    for (int i = 0; i < 16; ++i) {
        std::stringstream ss;
        ss << std::bitset<4>(i);
        next_action_range.push_back(ss.str());
    }

    int idx = 0;
    for (auto &token : coord_range)
        int_vocab[token] = idx++;
    for (auto &token : actions_range)
        str_vocab[std::string(1, token)] = idx++;
    for (auto &token : next_action_range)
        str_vocab[token] = idx++;
    str_vocab["!"] = idx;
}

std::vector<int> Encoder::encode(const std::vector<AgentsInfo> &agents,
                                 const std::vector<std::vector<int>> &cost2go) {
    std::vector<int> agents_indices;
    for (const auto &agent : agents) {
        agents_indices.push_back(int_vocab.at(agent.relative_pos.first));
        agents_indices.push_back(int_vocab.at(agent.relative_pos.second));
        agents_indices.push_back(int_vocab.at(
            std::clamp(agent.relative_goal.first,
                       -cfg.cost2go_value_limit, cfg.cost2go_value_limit)));
        agents_indices.push_back(int_vocab.at(
            std::clamp(agent.relative_goal.second,
                       -cfg.cost2go_value_limit, cfg.cost2go_value_limit)));
        for (const auto &action : agent.previous_actions)
            agents_indices.push_back(str_vocab.at(action));
        agents_indices.push_back(str_vocab.at(agent.next_action));
    }

    if (agents.size() < static_cast<size_t>(cfg.num_agents))
        agents_indices.insert(agents_indices.end(),
            (cfg.num_agents - agents.size()) * (5 + cfg.num_previous_actions),
            str_vocab["!"]);

    std::vector<int> cost2go_indices;
    for (const auto &row : cost2go)
        for (int value : row)
            cost2go_indices.push_back(int_vocab.at(value));

    std::vector<int> result;
    result.insert(result.end(), cost2go_indices.begin(), cost2go_indices.end());
    result.insert(result.end(), agents_indices.begin(), agents_indices.end());
    while (result.size() < static_cast<size_t>(cfg.context_size))
        result.push_back(str_vocab["!"]);
    if (cfg.task_type_id != -1)
        result.back() = cfg.task_type_id;
    return result;
}

// ---------------------------------------------------------------------------
// ObservationGenerator implementation
// ---------------------------------------------------------------------------

void ObservationGenerator::compute_agent_cost2go(int agent_idx) {
    const auto& goal = agents[agent_idx].goal;
    const int H = static_cast<int>(grid.size());
    const int W = static_cast<int>(grid[0].size());
    auto& cost = agent_cost2go[agent_idx];
    cost.assign(H, std::vector<uint16_t>(W, std::numeric_limits<uint16_t>::max()));

    if (goal.first < 0 || goal.first >= H || goal.second < 0 || goal.second >= W
        || grid[goal.first][goal.second] != 0)
        return;

    std::queue<std::pair<int,int>> fringe;
    fringe.push(goal);
    cost[goal.first][goal.second] = 0;
    const std::vector<std::pair<int,int>> moves = {{-1,0},{1,0},{0,-1},{0,1}};

    while (!fringe.empty()) {
        auto [r, c] = fringe.front();
        fringe.pop();
        uint16_t d = cost[r][c];
        for (const auto& move : moves) {
            int nr = r + move.first;
            int nc = c + move.second;
            if (nr >= 0 && nr < H && nc >= 0 && nc < W &&
                grid[nr][nc] == 0 &&
                cost[nr][nc] == std::numeric_limits<uint16_t>::max()) {
                cost[nr][nc] = d + 1;
                fringe.push({nr, nc});
            }
        }
    }
}

void ObservationGenerator::generate_cost2go_obs(int agent_idx,
                                                 std::vector<std::vector<int>> &buffer) {
    const auto& cost = agent_cost2go[agent_idx];
    const int H = static_cast<int>(cost.size());
    const int W = H > 0 ? static_cast<int>(cost[0].size()) : 0;
    const int r0 = agents[agent_idx].pos.first;
    const int c0 = agents[agent_idx].pos.second;

    auto safe_cost = [&](int ri, int cj) -> uint16_t {
        if (ri < 0 || ri >= H || cj < 0 || cj >= W)
            return std::numeric_limits<uint16_t>::max();
        return cost[static_cast<size_t>(ri)][static_cast<size_t>(cj)];
    };

    int middle_value = static_cast<int>(safe_cost(r0, c0));
    if (middle_value == static_cast<int>(std::numeric_limits<uint16_t>::max()))
        middle_value = 0;

    for (int i = 0; i <= cfg.obs_radius * 2; i++) {
        for (int j = 0; j <= cfg.obs_radius * 2; j++) {
            int ri = r0 - cfg.obs_radius + i;
            int cj = c0 - cfg.obs_radius + j;
            uint16_t raw = safe_cost(ri, cj);
            if (raw != std::numeric_limits<uint16_t>::max()) {
                int value = static_cast<int>(raw) - middle_value;
                if (value > cfg.cost2go_value_limit)
                    buffer[i][j] = cfg.cost2go_value_limit * 2;
                else if (value < -cfg.cost2go_value_limit)
                    buffer[i][j] = -cfg.cost2go_value_limit * 2;
                else
                    buffer[i][j] = value;
            } else {
                buffer[i][j] = -cfg.cost2go_value_limit * 4;
            }
        }
    }
}

int ObservationGenerator::get_distance(int agent_idx, const std::pair<int,int> &pos) {
    const auto& cost = agent_cost2go[agent_idx];
    const int H = static_cast<int>(cost.size());
    const int W = H > 0 ? static_cast<int>(cost[0].size()) : 0;
    if (pos.first < 0 || pos.first >= H || pos.second < 0 || pos.second >= W)
        return -1;
    uint16_t val = cost[static_cast<size_t>(pos.first)][static_cast<size_t>(pos.second)];
    if (val == std::numeric_limits<uint16_t>::max())
        return -1;
    return static_cast<int>(val);
}

void ObservationGenerator::create_agents(const std::vector<std::pair<int,int>> &positions,
                                          const std::vector<std::pair<int,int>> &goals) {
    agents.clear();
    int total_agents = static_cast<int>(positions.size());
    agents.resize(total_agents);
    cost2go_obs_buffer.resize(total_agents,
        std::vector<std::vector<int>>(2 * cfg.obs_radius + 1,
                                      std::vector<int>(2 * cfg.obs_radius + 1)));
    agent_cost2go.resize(total_agents,
        std::vector<std::vector<uint16_t>>(grid.size(),
            std::vector<uint16_t>(grid[0].size(), std::numeric_limits<uint16_t>::max())));
    for (int i = 0; i < total_agents; i++) {
        agents[i].pos = positions[i];
        agents[i].goal = goals[i];
        for (int j = 0; j < cfg.num_previous_actions; ++j)
            agents[i].action_history.push_back("n");
        compute_agent_cost2go(i);
        update_next_action(i);
    }
}

void ObservationGenerator::update_next_action(int agent_idx) {
    std::string next_action;
    auto &agent = agents[agent_idx];
    std::vector<std::pair<int,int>> moves = {{-1,0},{1,0},{0,-1},{0,1}};
    int current_cost = get_distance(agent_idx, agent.pos);

    for (const auto &move : moves) {
        std::pair<int,int> new_pos = {agent.pos.first + move.first,
                                       agent.pos.second + move.second};
        int neighbor_cost = get_distance(agent_idx, new_pos);
        if (neighbor_cost >= 0 && current_cost > neighbor_cost)
            next_action += "1";
        else
            next_action += "0";
    }
    agent.next_action = next_action;
}

static std::string mg_pogema_action_to_string(int a) {
    switch (a) {
    case 0: return "w";
    case 1: return "u";
    case 2: return "d";
    case 3: return "l";
    case 4: return "r";
    default: return "n";
    }
}

void ObservationGenerator::update_agents(const std::vector<std::pair<int,int>> &positions,
                                          const std::vector<std::pair<int,int>> &goals,
                                          const std::vector<int> &actions) {
    for (const auto &agent : agents)
        agents_locations[agent.pos.first][agent.pos.second] = -1;
    std::vector<size_t> need_to_update;
    for (size_t i = 0; i < agents.size(); i++) {
        auto &agent = agents[i];
        agents_locations[positions[i].first][positions[i].second] = static_cast<int>(i);
        agent.pos = positions[i];
        agent.action_history.push_back(mg_pogema_action_to_string(actions[i]));
        agent.action_history.pop_front();
        if (agent.goal != goals[i]) {
            agent.goal = goals[i];
            need_to_update.push_back(i);
        }
    }
    for (size_t i = 0; i < need_to_update.size(); i++)
        compute_agent_cost2go(need_to_update[i]);
    for (size_t i = 0; i < agents.size(); i++)
        update_next_action(i);
}

void ObservationGenerator::update_agents(const std::vector<std::pair<int,int>> &positions,
                                          const std::vector<std::pair<int,int>> &goals,
                                          const std::vector<std::array<int,5>> &action_histories) {
    for (const auto &agent : agents)
        agents_locations[agent.pos.first][agent.pos.second] = -1;
    std::vector<size_t> need_to_update;
    for (size_t i = 0; i < agents.size(); i++) {
        auto &agent = agents[i];
        agents_locations[positions[i].first][positions[i].second] = static_cast<int>(i);
        agent.pos = positions[i];
        agent.action_history.clear();
        for (int k = 0; k < 5; k++)
            agent.action_history.push_back(mg_pogema_action_to_string(action_histories[i][k]));
        if (agent.goal != goals[i]) {
            agent.goal = goals[i];
            need_to_update.push_back(i);
        }
    }
    for (size_t i = 0; i < need_to_update.size(); i++)
        compute_agent_cost2go(need_to_update[i]);
    for (size_t i = 0; i < agents.size(); i++)
        update_next_action(i);
}

std::vector<AgentsInfo> ObservationGenerator::get_agents_info(int agent_idx) {
    std::vector<AgentsInfo> agents_info;
    std::vector<int> considered_agents;
    const auto &cur_agent = agents[agent_idx];
    const int rows = static_cast<int>(agents_locations.size());
    const int cols = rows > 0 ? static_cast<int>(agents_locations[0].size()) : 0;
    for (int i = -cfg.agents_radius; i <= cfg.agents_radius; i++) {
        for (int j = -cfg.agents_radius; j <= cfg.agents_radius; j++) {
            int ni = cur_agent.pos.first + i;
            int nj = cur_agent.pos.second + j;
            if (ni < 0 || nj < 0 || ni >= rows || nj >= cols)
                continue;
            if (agents_locations[static_cast<size_t>(ni)][static_cast<size_t>(nj)] >= 0)
                considered_agents.push_back(
                    agents_locations[static_cast<size_t>(ni)][static_cast<size_t>(nj)]);
        }
    }
    std::vector<int> distances(considered_agents.size(), -1);
    for (size_t i = 0; i < considered_agents.size(); i++)
        distances[i] = std::abs(agents[considered_agents[i]].pos.first - cur_agent.pos.first) +
                       std::abs(agents[considered_agents[i]].pos.second - cur_agent.pos.second);
    std::vector<std::pair<int,int>> distance_agent_pairs;
    for (size_t i = 0; i < considered_agents.size(); i++)
        distance_agent_pairs.push_back({distances[i], considered_agents[i]});
    std::sort(distance_agent_pairs.begin(), distance_agent_pairs.end());

    for (int i = 0; i < std::min(static_cast<int>(distance_agent_pairs.size()), cfg.num_agents); i++) {
        const auto &agent = agents[distance_agent_pairs[i].second];
        agents_info.push_back(AgentsInfo(
            std::make_pair(agent.pos.first - cur_agent.pos.first,
                           agent.pos.second - cur_agent.pos.second),
            std::make_pair(agent.goal.first - cur_agent.pos.first,
                           agent.goal.second - cur_agent.pos.second),
            agent.action_history, agent.next_action));
    }
    return agents_info;
}

std::vector<std::vector<int>> ObservationGenerator::generate_observations() {
    std::vector<std::vector<int>> observations(agents.size());
    for (size_t i = 0; i < agents.size(); i++) {
        generate_cost2go_obs(i, cost2go_obs_buffer[i]);
        std::vector<AgentsInfo> agents_info = get_agents_info(i);
        observations[i] = encoder.encode(agents_info, cost2go_obs_buffer[i]);
    }
    return observations;
}

// ---------------------------------------------------------------------------
// TorchScript model wrapper — single input (observations only, no chat ids)
// ---------------------------------------------------------------------------

class MAPFGPTTorchscriptModel {
public:
    MAPFGPTTorchscriptModel(const std::string& model_path,
                             const std::string& device = "cpu");

    // observations: [B, N, T] -> returns [N, 5]
    torch::Tensor action_probs(const torch::Tensor& observations);

    // observations as std vectors [N][T] -> returns [N][5]
    std::vector<std::vector<float>> action_probs_from_vectors_std(
        const std::vector<std::vector<int64_t>>& observations);

private:
    torch::jit::script::Module module_;
    torch::Device device_;

    // AOTI mode: directory of n<N>_bf16.pt2 packages, dispatch by N + zero-pad.
    bool aoti_mode_ = false;
    std::vector<std::pair<int, std::unique_ptr<torch::inductor::AOTIModelPackageLoader>>> aoti_loaders_;

    static torch::Tensor vector2d_to_long_tensor(
        const std::vector<std::vector<int64_t>>& data);
    static torch::Device parse_device(const std::string& device_name);
};

MAPFGPTTorchscriptModel::MAPFGPTTorchscriptModel(const std::string& model_path,
                                                     const std::string& device)
    : device_(parse_device(device)) {
    // Detect mode: .pt2 file → AOTI dynamic; directory → AOTI fixed-N dispatch;
    // .pt file → legacy TorchScript.
    namespace fs = std::filesystem;
    if (fs::exists(model_path) && fs::is_regular_file(model_path)
        && model_path.size() >= 4
        && model_path.compare(model_path.size() - 4, 4, ".pt2") == 0) {
        if (!device_.is_cuda()) {
            throw std::runtime_error(
                "MAPF-GPT AOTI .pt2 mode requires CUDA, got: " + device);
        }
        std::cerr << "[mapf-gpt] AOTI dynamic loading " << model_path << std::endl;
        aoti_loaders_.emplace_back(0,
            std::make_unique<torch::inductor::AOTIModelPackageLoader>(model_path));
        aoti_mode_ = true;
        std::cerr << "[mapf-gpt] AOTI dynamic-N mode active" << std::endl;
        return;
    }
    if (fs::exists(model_path) && fs::is_directory(model_path)) {
        if (!device_.is_cuda()) {
            throw std::runtime_error(
                "MAPF-GPT AOTI mode requires CUDA, got: " + device);
        }
        std::regex pt2_re(R"(n(\d+)_bf16\.pt2)");
        for (const auto& entry : fs::directory_iterator(model_path)) {
            std::smatch m;
            std::string fname = entry.path().filename().string();
            if (std::regex_match(fname, m, pt2_re)) {
                int n = std::stoi(m[1].str());
                std::cerr << "[mapf-gpt] AOTI: loading n=" << n << " from " << entry.path() << std::endl;
                aoti_loaders_.emplace_back(n,
                    std::make_unique<torch::inductor::AOTIModelPackageLoader>(entry.path().string()));
            }
        }
        if (aoti_loaders_.empty()) {
            throw std::runtime_error(
                "MAPF-GPT AOTI: no n<N>_bf16.pt2 found in " + model_path);
        }
        std::sort(aoti_loaders_.begin(), aoti_loaders_.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        aoti_mode_ = true;
        std::cerr << "[mapf-gpt] AOTI mode active with " << aoti_loaders_.size()
                  << " packages, max N=" << aoti_loaders_.back().first << std::endl;
        return;
    }

    module_ = torch::jit::load(model_path);
    module_.to(device_);
    module_.eval();
}

torch::Device MAPFGPTTorchscriptModel::parse_device(const std::string& device_name) {
    if (device_name == "cpu")
        return torch::Device(torch::kCPU);
    if (device_name.rfind("cuda", 0) == 0)
        return torch::Device(device_name);
    if (device_name == "mps")
        return torch::Device(torch::kMPS);
    throw std::invalid_argument("Unsupported device string: " + device_name);
}

torch::Tensor MAPFGPTTorchscriptModel::vector2d_to_long_tensor(
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

torch::Tensor MAPFGPTTorchscriptModel::action_probs(const torch::Tensor& observations) {
    if (observations.dim() != 3)
        throw std::invalid_argument("observations must be rank-3 [B, N, T]");

    auto obs = observations.to(device_, torch::kLong, false, false);

    if (aoti_mode_) {
        c10::InferenceMode guard(true);
        const int64_t actual_n = obs.size(1);

        // Dynamic-N single-package: no dispatch.
        if (aoti_loaders_.size() == 1 && aoti_loaders_.front().first == 0) {
            auto outs = aoti_loaders_.front().second->run({obs});
            if (outs.empty()) {
                throw std::runtime_error("MAPF-GPT AOTI dynamic: runner returned no outputs");
            }
            return outs[0];
        }

        // Fixed-N dispatch mode.
        auto it = std::lower_bound(
            aoti_loaders_.begin(), aoti_loaders_.end(), actual_n,
            [](const auto& p, int64_t n) { return p.first < n; });
        if (it == aoti_loaders_.end()) {
            throw std::runtime_error(
                "MAPF-GPT AOTI: no .pt2 for N=" + std::to_string(actual_n)
                + " (max compiled N=" + std::to_string(aoti_loaders_.back().first) + ")");
        }
        const int64_t target_n = it->first;
        torch::Tensor obs_in = obs;
        if (actual_n < target_n) {
            const int64_t pad = target_n - actual_n;
            auto pad_obs = torch::zeros({obs.size(0), pad, obs.size(2)},
                torch::TensorOptions().dtype(obs.dtype()).device(obs.device()));
            obs_in = torch::cat({obs, pad_obs}, /*dim=*/1);
        }
        auto outs = it->second->run({obs_in});
        if (outs.empty()) {
            throw std::runtime_error("MAPF-GPT AOTI: runner returned no outputs");
        }
        torch::Tensor result = outs[0];
        if (actual_n < target_n) {
            result = result.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_n);
        }
        return result;
    }

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(obs);
    return module_.forward(inputs).toTensor();
}

std::vector<std::vector<float>> MAPFGPTTorchscriptModel::action_probs_from_vectors_std(
    const std::vector<std::vector<int64_t>>& observations) {
    auto obs = vector2d_to_long_tensor(observations).unsqueeze(0);  // [1, N, T]
    auto probs = action_probs(obs).to(torch::kCPU).contiguous();
    if (probs.dim() != 2)
        throw std::runtime_error("Expected 2D probability tensor [N, 5]");

    const int64_t rows = probs.size(0);
    const int64_t cols = probs.size(1);
    const float* data = probs.data_ptr<float>();
    std::vector<std::vector<float>> out(static_cast<size_t>(rows),
                                         std::vector<float>(static_cast<size_t>(cols)));
    for (int64_t i = 0; i < rows; ++i)
        for (int64_t j = 0; j < cols; ++j)
            out[static_cast<size_t>(i)][static_cast<size_t>(j)] = data[i * cols + j];
    return out;
}

}  // namespace mapf_gpt

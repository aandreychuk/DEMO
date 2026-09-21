// Standalone C++ LC-MAPF core: observation generator + TorchScript model.
#include <torch/script.h>
#include <torch/csrc/jit/passes/freeze_module.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#if __has_include(<cuda_runtime_api.h>)
#define LAGAT_LC_HAS_CUDA_HEADERS 1
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAGraph.h>
#else
#define LAGAT_LC_HAS_CUDA_HEADERS 0
#endif
#include <ATen/autocast_mode.h>
#include <filesystem>
#include <regex>
#include <array>
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <deque>
#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <bitset>
#include <algorithm>
#include <sstream>
#include <limits>
#include <cstdint>
#include <stdexcept>

struct InputParameters
{
    InputParameters(int cvl = 20, int na = 13, int npa = 5, int cs = 256, int obsr = 5, int ar = 5, int tt = -1) : cost2go_value_limit(cvl),
                                                                                                                      num_agents(na),
                                                                                                                      num_previous_actions(npa),
                                                                                                                      context_size(cs),
                                                                                                                      obs_radius(obsr),
                                                                                                                      agents_radius(ar),
                                                                                                                      task_type_id(tt) {}
    int cost2go_value_limit;
    int num_agents;
    int num_previous_actions;
    int context_size;
    int obs_radius;
    int agents_radius;
    int task_type_id;
};

struct HashPair
{
    uint64_t operator()(const std::pair<int, int>& p) const {
        return (uint64_t(p.first) << 32) | uint64_t(p.second);
    }
};

struct AgentsInfo
{
    AgentsInfo(std::pair<int, int> rp, std::pair<int, int> rg, std::deque<std::string> pa, std::string na) : relative_pos(rp), relative_goal(rg), previous_actions(pa), next_action(na) {}
    AgentsInfo() {}
    std::pair<int, int> relative_pos;
    std::pair<int, int> relative_goal;
    std::deque<std::string> previous_actions;
    std::string next_action;
};

struct Agent
{
    std::pair<int, int> pos;
    std::pair<int, int> goal;
    std::deque<std::string> action_history;
    std::string next_action;
};

class Encoder
{
public:
    InputParameters cfg;
    std::vector<int> coord_range;
    std::vector<char> actions_range;
    std::vector<std::string> next_action_range;
    std::unordered_map<std::string, int> str_vocab;
    std::unordered_map<int, int> int_vocab;
    std::unordered_map<int, int> inverse_int_vocab;
    std::unordered_map<int, std::string> inverse_str_vocab;
    Encoder(const InputParameters &cfg);
    std::vector<int> encode(const std::vector<AgentsInfo> &agents, const std::vector<std::vector<int>> &cost2go);
    std::pair<std::vector<AgentsInfo>, std::vector<std::vector<int>>> decode(const std::vector<int> &observation);
};

class VectorObservation
{
public:
    InputParameters cfg;
    VectorObservation(const InputParameters &cfg);
    std::vector<float> vectorize(const std::vector<AgentsInfo> &agents, const std::vector<std::vector<int>> &cost2go);
};

class ObservationGenerator
{
public:
    std::vector<Agent> agents;
    InputParameters cfg;
    Encoder encoder;
    VectorObservation vector_observation;
    std::vector<std::vector<int>> agents_locations;
    std::vector<std::vector<std::vector<int>>> cost2go_obs_buffer; // Buffer for each agent
    std::vector<std::vector<int>> grid;
    std::vector<std::vector<int>> agents_in_obs;
    // Full-map cost2go per agent: agent_cost2go[agent_id][row][col] = distance from goal (or max if unreachable)
    std::vector<std::vector<std::vector<uint16_t>>> agent_cost2go;
    // Optional per-agent dynamic obstacles. Lifelong warehouse agents without
    // a pallet leave this empty; loaded agents mark currently parked pallets.
    std::vector<std::vector<std::vector<uint8_t>>> dynamic_obstacles;
    ObservationGenerator(const std::vector<std::vector<int>> &grid, const InputParameters &cfg)
        : grid(grid), cfg(cfg), encoder(cfg), vector_observation(cfg)
    {
        agents_locations = std::vector<std::vector<int>>(grid.size(), std::vector<int>(grid[0].size(), -1));
    }
    ~ObservationGenerator() {}
    void compute_agent_cost2go(int agent_idx);
    void generate_cost2go_obs(int agent_idx, bool only_obstacles, std::vector<std::vector<int>> &buffer);
    int get_distance(int agent_idx, const std::pair<int, int> &pos);
    void create_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals);
    void set_dynamic_obstacles(const std::vector<std::vector<std::pair<int, int>>> &cells);
    void update_next_action(int agent_idx);
    void update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<int> &actions);
    // Set positions, goals, and full 5-action history (POGEMA: 0=wait, 1=up, 2=down, 3=left, 4=right, -1=no action). Used when restoring state from LaCAM HNode.
    void update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<std::array<int, 5>> &action_histories);
    std::vector<AgentsInfo> get_agents_info(int agent_idx);
    std::vector<std::vector<int>> generate_observations();
    std::vector<std::vector<float>> generate_vector_observations();
    std::vector<std::vector<float>> encoded_to_vector_observations(const std::vector<std::vector<int>> &observations);
    std::vector<std::vector<int>> get_agents_in_obs();
};

// cppimport

void ObservationGenerator::compute_agent_cost2go(int agent_idx)
{
    const auto& goal = agents[agent_idx].goal;
    const int H = static_cast<int>(grid.size());
    const int W = static_cast<int>(grid[0].size());
    auto& cost = agent_cost2go[agent_idx];
    cost.assign(H, std::vector<uint16_t>(W, std::numeric_limits<uint16_t>::max()));

    auto blocked = [&](int row, int col) {
        return grid[row][col] != 0 ||
               (!dynamic_obstacles.empty() && dynamic_obstacles[agent_idx][row][col] != 0);
    };

    if (goal.first < 0 || goal.first >= H || goal.second < 0 || goal.second >= W || blocked(goal.first, goal.second))
        return;

    std::queue<std::pair<int, int>> fringe;
    fringe.push(goal);
    cost[goal.first][goal.second] = 0;
    const std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    while (!fringe.empty())
    {
        auto [r, c] = fringe.front();
        fringe.pop();
        uint16_t d = cost[r][c];
        for (const auto& move : moves)
        {
            int nr = r + move.first;
            int nc = c + move.second;
            if (nr >= 0 && nr < H && nc >= 0 && nc < W &&
                !blocked(nr, nc) && cost[nr][nc] == std::numeric_limits<uint16_t>::max())
            {
                cost[nr][nc] = d + 1;
                fringe.push({nr, nc});
            }
        }
    }
}

void ObservationGenerator::generate_cost2go_obs(int agent_idx, bool only_obstacles, std::vector<std::vector<int>> &buffer)
{
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

    if (only_obstacles) {
        for (int i = 0; i <= cfg.obs_radius * 2; i++) {
            for (int j = 0; j <= cfg.obs_radius * 2; j++) {
                int ri = r0 - cfg.obs_radius + i;
                int cj = c0 - cfg.obs_radius + j;
                uint16_t val = safe_cost(ri, cj);
                buffer[i][j] = (val == std::numeric_limits<uint16_t>::max()) ? 1 : 0;
            }
        }
        return;
    }

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

int ObservationGenerator::get_distance(int agent_idx, const std::pair<int, int> &pos)
{
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

VectorObservation::VectorObservation(const InputParameters &cfg) : cfg(cfg)
{
    return;
}

std::vector<float> VectorObservation::vectorize(const std::vector<AgentsInfo> &agents, const std::vector<std::vector<int>> &cost2go)
{
    std::vector<float> result(256, 0);
    static int obstacle_value = -cfg.cost2go_value_limit * 4;
    static float clamp_value = 0.9;
    int cur_idx = 0;
    for (const auto &row : cost2go)
        for (int value : row)
        {
            if(value == obstacle_value)
                result[cur_idx] = 1;
            else if(value > cfg.cost2go_value_limit)
                result[cur_idx] = clamp_value;
            else if(value < - cfg.cost2go_value_limit)
                result[cur_idx] = -clamp_value;
            else
                result[cur_idx] = float(value)/(cfg.cost2go_value_limit + 5);
            cur_idx++;
        }
    for (const auto &agent : agents)
    {
        result[cur_idx] = float(agent.relative_pos.first)/cfg.cost2go_value_limit;
        result[cur_idx + 1] = float(agent.relative_pos.second)/cfg.cost2go_value_limit;
        result[cur_idx + 2] = float(std::clamp(agent.relative_goal.first, -cfg.cost2go_value_limit, cfg.cost2go_value_limit))/cfg.cost2go_value_limit;
        result[cur_idx + 3] = float(std::clamp(agent.relative_goal.second, -cfg.cost2go_value_limit, cfg.cost2go_value_limit))/cfg.cost2go_value_limit;
        cur_idx += 4;
        for (const auto &action : agent.previous_actions)
        {
            if(action == "n")
                result[cur_idx] = 0.0;
            else if(action == "w")
                result[cur_idx] = 0.1;
            else if(action == "u")
                result[cur_idx] = 0.2;
            else if(action == "d")
                result[cur_idx] = 0.3;
            else if(action == "l")
                result[cur_idx] = 0.4;
            else if(action == "r")
                result[cur_idx] = 0.5;
            cur_idx++;
        }
        result[cur_idx] = (int(agent.next_action.at(0) == '1')*8 + int(agent.next_action.at(1) == '1')*4 + int(agent.next_action.at(2) == '1')*2 + int(agent.next_action.at(3) == '1') + 1)/16.0;
        cur_idx++;
    }
    return result;
}

Encoder::Encoder(const InputParameters &cfg) : cfg(cfg)
{
    for (int i = -cfg.cost2go_value_limit; i <= cfg.cost2go_value_limit; ++i)
        coord_range.push_back(i);
    coord_range.push_back(-cfg.cost2go_value_limit * 4);
    coord_range.push_back(-cfg.cost2go_value_limit * 2);
    coord_range.push_back(cfg.cost2go_value_limit * 2);

    actions_range = {'n', 'w', 'u', 'd', 'l', 'r'};
    for (int i = 0; i < 16; ++i)
    {
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

    for (auto &[token, idx] : int_vocab)
        inverse_int_vocab[idx] = token;
    for (auto &[token, idx] : str_vocab)
        inverse_str_vocab[idx] = token;

}

std::vector<int> Encoder::encode(const std::vector<AgentsInfo> &agents, const std::vector<std::vector<int>> &cost2go)
{
    std::vector<int> agents_indices;
    for (const auto &agent : agents)
    {
        std::vector<int> coord_indices = {
            int_vocab.at(agent.relative_pos.first),
            int_vocab.at(agent.relative_pos.second),
            int_vocab.at(std::clamp(agent.relative_goal.first, -cfg.cost2go_value_limit, cfg.cost2go_value_limit)),
            int_vocab.at(std::clamp(agent.relative_goal.second, -cfg.cost2go_value_limit, cfg.cost2go_value_limit))};

        std::vector<int> actions_indices;
        for (const auto &action : agent.previous_actions)
        {
            actions_indices.push_back(str_vocab.at(action));
        }
        std::vector<int> next_action_indices = {str_vocab.at(agent.next_action)};

        agents_indices.insert(agents_indices.end(), coord_indices.begin(), coord_indices.end());
        agents_indices.insert(agents_indices.end(), actions_indices.begin(), actions_indices.end());
        agents_indices.insert(agents_indices.end(), next_action_indices.begin(), next_action_indices.end());
    }

    if (agents.size() < static_cast<size_t>(cfg.num_agents))
        agents_indices.insert(
            agents_indices.end(),
            (cfg.num_agents - agents.size()) * (5 + cfg.num_previous_actions),
            str_vocab.at("!"));

    std::vector<int> cost2go_indices;
    for (const auto &row : cost2go)
        for (int value : row)
            cost2go_indices.push_back(int_vocab.at(value));

    std::vector<int> result;
    result.insert(result.end(), cost2go_indices.begin(), cost2go_indices.end());
    result.insert(result.end(), agents_indices.begin(), agents_indices.end());
    while (result.size() < 256)
        result.push_back(str_vocab.at("!"));
    if (cfg.task_type_id != -1)
        result.back() = cfg.task_type_id;
    return result;
}

std::pair<std::vector<AgentsInfo>, std::vector<std::vector<int>>> Encoder::decode(const std::vector<int> &observation)
{
    std::vector<AgentsInfo> agents;
    int grid_size = (2 * cfg.obs_radius + 1);
    std::vector<std::vector<int>> cost2go(grid_size, std::vector<int>(grid_size));

    // Decode cost2go grid
    int idx = 0;
    for (int i = 0; i < grid_size; i++) {
        for (int j = 0; j < grid_size; j++) {
            cost2go[i][j] = inverse_int_vocab.at(observation[idx++]);
        }
    }

    // Decode agents information
    int agent_data_size = 5 + cfg.num_previous_actions;
    for (int agent = 0; agent < cfg.num_agents; agent++) {
        int base_idx = grid_size * grid_size + agent * agent_data_size;

        // Check if this is a padding agent
        if (inverse_str_vocab.find(observation[base_idx]) != inverse_str_vocab.end()) {
            if (inverse_str_vocab.at(observation[base_idx]) == "!") {
                break;
            }
        }

        // Decode relative positions and goals
        std::pair<int, int> relative_pos = {
            inverse_int_vocab.at(observation[base_idx]),
            inverse_int_vocab.at(observation[base_idx + 1])
        };
        std::pair<int, int> relative_goal = {
            inverse_int_vocab.at(observation[base_idx + 2]),
            inverse_int_vocab.at(observation[base_idx + 3])
        };

        // Decode previous actions
        std::deque<std::string> previous_actions;
        for (int i = 0; i < cfg.num_previous_actions; i++) {
            previous_actions.push_back(
                inverse_str_vocab.at(observation[base_idx + 4 + i])
            );
        }

        // Decode next action
        std::string next_action = inverse_str_vocab.at(
            observation[base_idx + 4 + cfg.num_previous_actions]
        );

        // Create AgentInfo object and add to vector
        agents.push_back(AgentsInfo(
            relative_pos,
            relative_goal,
            previous_actions,
            next_action
        ));
    }

    return {agents, cost2go};
}

std::vector<std::vector<float>> ObservationGenerator::encoded_to_vector_observations(const std::vector<std::vector<int>> &observations)
{
    std::vector<std::vector<float>> result;
    for (const auto &observation : observations)
    {
        std::vector<AgentsInfo> agents;
        std::vector<std::vector<int>> cost2go;
        std::tie(agents, cost2go) = encoder.decode(observation);
        try
        {
            result.push_back(vector_observation.vectorize(agents, cost2go));
        }
        catch (const std::exception &e)
        {
            std::cerr << "Error vectorizing observation: " << e.what() << std::endl;
            result.push_back(std::vector<float>(256, 0));
        }
    }
    return result;
}

void ObservationGenerator::create_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals)
{
    agents.clear();
    int total_agents = static_cast<int>(positions.size());
    agents.resize(total_agents);
    agents_in_obs.resize(total_agents);
    cost2go_obs_buffer.resize(total_agents, std::vector<std::vector<int>>(2 * cfg.obs_radius + 1, std::vector<int>(2 * cfg.obs_radius + 1)));
    agent_cost2go.resize(total_agents, std::vector<std::vector<uint16_t>>(grid.size(), std::vector<uint16_t>(grid[0].size(), std::numeric_limits<uint16_t>::max())));
    dynamic_obstacles.resize(total_agents, std::vector<std::vector<uint8_t>>(grid.size(), std::vector<uint8_t>(grid[0].size(), 0)));
    for (int i = 0; i < total_agents; i++)
    {
        agents[i].pos = positions[i];
        agents[i].goal = goals[i];
        for (int j = 0; j < cfg.num_previous_actions; ++j)
            agents[i].action_history.push_back("n");
        compute_agent_cost2go(i);
        update_next_action(i);
    }
}

void ObservationGenerator::set_dynamic_obstacles(const std::vector<std::vector<std::pair<int, int>>> &cells)
{
    if (cells.size() != agents.size())
        throw std::runtime_error("dynamic obstacle rows must match agent count");
    const int H = static_cast<int>(grid.size());
    const int W = H > 0 ? static_cast<int>(grid[0].size()) : 0;
    for (size_t agent_idx = 0; agent_idx < agents.size(); ++agent_idx)
    {
        std::vector<std::vector<uint8_t>> next(H, std::vector<uint8_t>(W, 0));
        for (const auto &[row, col] : cells[agent_idx])
        {
            if (row >= 0 && row < H && col >= 0 && col < W)
                next[row][col] = 1;
        }
        if (next == dynamic_obstacles[agent_idx])
            continue;
        dynamic_obstacles[agent_idx] = std::move(next);
        compute_agent_cost2go(static_cast<int>(agent_idx));
        update_next_action(static_cast<int>(agent_idx));
    }
}

void ObservationGenerator::update_next_action(int agent_idx)
{
    std::string next_action;
    auto &agent = agents[agent_idx];
    std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int current_cost = get_distance(agent_idx, agent.pos);

    for (const auto &move : moves)
    {
        std::pair<int, int> new_pos = {agent.pos.first + move.first, agent.pos.second + move.second};
        int neighbor_cost = get_distance(agent_idx, new_pos);

        if (neighbor_cost >= 0 && current_cost > neighbor_cost)
            next_action += "1";
        else
            next_action += "0";
    }
    agent.next_action = next_action;
}

static std::string pogema_action_to_string(int a)
{
    switch (a) {
    case 0: return "w";
    case 1: return "u";
    case 2: return "d";
    case 3: return "l";
    case 4: return "r";
    default: return "n";
    }
}

void ObservationGenerator::update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<int> &actions)
{
    for (const auto &agent : agents)
        agents_locations[agent.pos.first][agent.pos.second] = -1; // first clear old locations for ALL agents
    std::vector<size_t> need_to_update;
    for (size_t i = 0; i < agents.size(); i++)
    {
        auto &agent = agents[i];
        agents_locations[positions[i].first][positions[i].second] = i;
        agent.pos = positions[i];
        agent.action_history.push_back(pogema_action_to_string(actions[i]));
        agent.action_history.pop_front();
        if (agent.goal != goals[i])
        {
            agent.goal = goals[i];
            need_to_update.push_back(i);
        }
    }

    for (size_t i = 0; i < need_to_update.size(); i++)
        compute_agent_cost2go(need_to_update[i]);
    for (size_t i = 0; i < agents.size(); i++)
        update_next_action(i);
}

void ObservationGenerator::update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<std::array<int, 5>> &action_histories)
{
    // Assume positions/goals/action_histories all have size == agents.size()
    for (const auto &agent : agents)
        agents_locations[agent.pos.first][agent.pos.second] = -1;

    std::vector<size_t> need_to_update;
    for (size_t i = 0; i < agents.size(); i++)
    {
        auto &agent = agents[i];
        agents_locations[positions[i].first][positions[i].second] = static_cast<int>(i);
        agent.pos = positions[i];
        agent.action_history.clear();
        for (int k = 0; k < 5; k++)
            agent.action_history.push_back(pogema_action_to_string(action_histories[i][k]));
        if (agent.goal != goals[i])
        {
            agent.goal = goals[i];
            need_to_update.push_back(i);
        }
    }
    for (size_t i = 0; i < need_to_update.size(); i++)
        compute_agent_cost2go(need_to_update[i]);
    for (size_t i = 0; i < agents.size(); i++)
        update_next_action(i);
}

std::vector<AgentsInfo> ObservationGenerator::get_agents_info(int agent_idx)
{
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
                considered_agents.push_back(agents_locations[static_cast<size_t>(ni)][static_cast<size_t>(nj)]);
        }
    }
    std::vector<int> distances(considered_agents.size(), -1);
    for (size_t i = 0; i < considered_agents.size(); i++)
        distances[i] = std::abs(agents[considered_agents[i]].pos.first - cur_agent.pos.first) +
                       std::abs(agents[considered_agents[i]].pos.second - cur_agent.pos.second);
    std::vector<std::pair<int, int>> distance_agent_pairs;
    for (size_t i = 0; i < considered_agents.size(); i++)
    {
        distance_agent_pairs.push_back({distances[i], considered_agents[i]});
    }
    std::sort(distance_agent_pairs.begin(), distance_agent_pairs.end());
    std::vector<int> agents_ids(cfg.num_agents, -1);
    for (int i = 0; i < std::min(int(distance_agent_pairs.size()), cfg.num_agents); i++)
    {
        const auto &agent = agents[distance_agent_pairs[i].second];
        agents_info.push_back(AgentsInfo(std::make_pair(agent.pos.first - cur_agent.pos.first, agent.pos.second - cur_agent.pos.second),
                                         std::make_pair(agent.goal.first - cur_agent.pos.first, agent.goal.second - cur_agent.pos.second),
                                         agent.action_history, agent.next_action));
        agents_ids[i] = distance_agent_pairs[i].second;
    }
    agents_in_obs[agent_idx] = agents_ids;
    return agents_info;
}

std::vector<std::vector<int>> ObservationGenerator::generate_observations()
{

    std::vector<std::vector<int>> observations(agents.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int64_t i = 0; i < static_cast<int64_t>(agents.size()); i++)
    {
        const auto index = static_cast<size_t>(i);
        generate_cost2go_obs(static_cast<int>(index), false,
                             cost2go_obs_buffer[index]);
        std::vector<AgentsInfo> agents_info =
            get_agents_info(static_cast<int>(index));
        observations[index] =
            encoder.encode(agents_info, cost2go_obs_buffer[index]);
    }
    return observations;
}

std::vector<std::vector<float>> ObservationGenerator::generate_vector_observations()
{
    std::vector<std::vector<float>> observations(agents.size());
    for (size_t i = 0; i < agents.size(); i++)
        observations[i] = vector_observation.vectorize(get_agents_info(i), cost2go_obs_buffer[i]);
    return observations;
}

std::vector<std::vector<int>> ObservationGenerator::get_agents_in_obs()
{
    return agents_in_obs;
}



namespace lc_mapf {

class LCMAPFTorchscriptModel {
public:
    LCMAPFTorchscriptModel(const std::string& model_path, const std::string& device = "cpu");

    // observations: [B, C, T], agent_chat_ids: [B, C, L]
    // Returns action probabilities with shape [B*C, 5] (matches Python get_action_probs output).
    torch::Tensor action_probs(const torch::Tensor& observations, const torch::Tensor& agent_chat_ids);

    // observations: [C, T], agent_chat_ids: [C, L] -> internally unsqueezed to batch size 1.
    torch::Tensor action_probs_from_vectors(
        const std::vector<std::vector<int64_t>>& observations,
        const std::vector<std::vector<int64_t>>& agent_chat_ids
    );
    std::vector<std::vector<float>> action_probs_from_vectors_std(
        const std::vector<std::vector<int64_t>>& observations,
        const std::vector<std::vector<int64_t>>& agent_chat_ids
    );

    // Greedy action (argmax over action probabilities) for each flattened agent row.
    std::vector<int64_t> greedy_actions(const torch::Tensor& observations, const torch::Tensor& agent_chat_ids);
    std::vector<int64_t> greedy_actions_from_vectors(
        const std::vector<std::vector<int64_t>>& observations,
        const std::vector<std::vector<int64_t>>& agent_chat_ids
    );

private:
    torch::jit::script::Module module_;
    torch::Device device_;

    // AOTI mode: when model_path is a directory of *.pt2 packages, we use
    // pre-compiled AOTInductor packages instead of TorchScript module.
    // Each package is compiled for a specific N (number of agents). At
    // inference time we dispatch on actual N: pick smallest pre-compiled
    // pre_n >= actual_n and pad inputs (zero) up to pre_n, then slice
    // outputs back to actual_n. AOTI bakes in BF16 autocast, fused kernels,
    // and constant-folded forward — typically 3-4× vs TS+Cuda Graph.
    bool aoti_mode_ = false;
    // Sorted ascending by N. key = N agents that the .pt2 was compiled for.
    std::vector<std::pair<int, std::unique_ptr<torch::inductor::AOTIModelPackageLoader>>> aoti_loaders_;

    // Autocast mode for mixed-precision inference (LAGAT_LC_AUTOCAST=bf16|fp16).
    enum class AutocastMode { None, BF16, FP16 };
    AutocastMode autocast_mode_ = AutocastMode::None;

    // CUDA Graph capture (LAGAT_LC_CUDA_GRAPH=1):
    // Captures forward pass once after warmup; subsequent calls use replay()
    // → eliminates per-call kernel launch overhead. Requires fixed input
    // shapes across calls (verified once on first call, fallback to regular
    // forward on shape mismatch).
    bool graph_enabled_ = false;
    bool graph_captured_ = false;
    int warmup_remaining_ = 3;
    torch::Tensor static_obs_;
    torch::Tensor static_chat_;
    torch::Tensor static_out_;
#if LAGAT_LC_HAS_CUDA_HEADERS
    std::unique_ptr<at::cuda::CUDAGraph> graph_;
#endif

    static torch::Tensor vector2d_to_long_tensor(const std::vector<std::vector<int64_t>>& data);
    static void check_batched_shapes(const torch::Tensor& observations, const torch::Tensor& agent_chat_ids);
};

}  // namespace lc_mapf



#include <stdexcept>

namespace lc_mapf {

namespace {

torch::Device parse_device(const std::string& device_name) {
    if (device_name == "cpu") {
        return torch::Device(torch::kCPU);
    }
    if (device_name.rfind("cuda", 0) == 0) {
        return torch::Device(device_name);
    }
    if (device_name == "mps") {
        return torch::Device(torch::kMPS);
    }
    throw std::invalid_argument("Unsupported device string: " + device_name);
}

}  // namespace

LCMAPFTorchscriptModel::LCMAPFTorchscriptModel(const std::string& model_path, const std::string& device)
    : device_(parse_device(device)) {
    // Detect AOTI mode:
    //  (1) model_path is a single .pt2 file → AOTI dynamic-N (single package
    //      handles any N; usually 5-10% slower than fixed but no padding/dispatch).
    //  (2) model_path is a directory containing n<N>_bf16.pt2 files → AOTI
    //      dispatch by N + zero-pad for unusual N (max-autotune fixed shape).
    //  (3) model_path is a single .pt file → legacy TorchScript.
    namespace fs = std::filesystem;
    if (fs::exists(model_path) && fs::is_regular_file(model_path)
        && model_path.size() >= 4
        && model_path.compare(model_path.size() - 4, 4, ".pt2") == 0) {
        if (!device_.is_cuda()) {
            throw std::runtime_error(
                "AOTI .pt2 mode requires CUDA, got: " + device);
        }
        std::cerr << "[lc-mapf] AOTI dynamic loading " << model_path << std::endl;
        // Dynamic-N package occupies the N=0 slot; lookup is direct.
        aoti_loaders_.emplace_back(0,
            std::make_unique<torch::inductor::AOTIModelPackageLoader>(model_path));
        aoti_mode_ = true;
        std::cerr << "[lc-mapf] AOTI dynamic-N mode active" << std::endl;
        return;
    }
    if (fs::exists(model_path) && fs::is_directory(model_path)) {
        if (!device_.is_cuda()) {
            throw std::runtime_error(
                "AOTI mode (directory model_path) requires CUDA device, got: " + device);
        }
        std::regex pt2_re(R"(n(\d+)_bf16\.pt2)");
        for (const auto& entry : fs::directory_iterator(model_path)) {
            std::smatch m;
            std::string fname = entry.path().filename().string();
            if (std::regex_match(fname, m, pt2_re)) {
                int n = std::stoi(m[1].str());
                std::cerr << "[lc-mapf] AOTI: loading n=" << n << " from " << entry.path() << std::endl;
                auto loader = std::make_unique<torch::inductor::AOTIModelPackageLoader>(
                    entry.path().string());
                aoti_loaders_.emplace_back(n, std::move(loader));
            }
        }
        if (aoti_loaders_.empty()) {
            throw std::runtime_error(
                "AOTI mode: no n<N>_bf16.pt2 files found under " + model_path);
        }
        std::sort(aoti_loaders_.begin(), aoti_loaders_.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        aoti_mode_ = true;
        std::cerr << "[lc-mapf] AOTI mode active with " << aoti_loaders_.size()
                  << " packages, max N=" << aoti_loaders_.back().first << std::endl;
        // BF16 autocast and CUDA Graph are baked into the AOTI graphs;
        // the legacy env-var paths below are no-ops in AOTI mode.
        return;
    }

    // Legacy TorchScript path
    module_ = torch::jit::load(model_path);
    module_.to(device_);
    module_.eval();
    // Mixed precision via autocast (LAGAT_LC_AUTOCAST=bf16|fp16):
    // Wrap forward in autocast — keeps weights FP32, ops auto-cast to half.
    // No risk of dtype mismatch like full conversion; H100 tensor cores
    // give ~2× compute at FP16/BF16. We just record the mode here, apply
    // in action_probs forward.
    autocast_mode_ = AutocastMode::None;
    if (device_.is_cuda()) {
        if (const char *e = std::getenv("LAGAT_LC_AUTOCAST")) {
            if (std::string(e) == "bf16") {
                autocast_mode_ = AutocastMode::BF16;
                std::cerr << "[lc-mapf] autocast BF16 enabled" << std::endl;
            } else if (std::string(e) == "fp16") {
                autocast_mode_ = AutocastMode::FP16;
                std::cerr << "[lc-mapf] autocast FP16 enabled" << std::endl;
            }
        }
    }
    // CUDA Graph capture (LAGAT_LC_CUDA_GRAPH=1): captures forward after
    // warmup, subsequent calls use replay() → eliminates per-call kernel
    // launch overhead (which is ~85ms/call on LC-MAPF, the dominant cost).
    if (device_.is_cuda() && std::getenv("LAGAT_LC_CUDA_GRAPH") != nullptr) {
#if LAGAT_LC_HAS_CUDA_HEADERS
        graph_enabled_ = true;
        std::cerr << "[lc-mapf] CUDA Graph capture ENABLED (warmup=3 calls)" << std::endl;
#else
        throw std::runtime_error(
            "LAGAT_LC_CUDA_GRAPH requires CUDA development headers");
#endif
    }
    // Inference-time graph optimizations:
    //   1. freeze: turn parameters into compile-time constants (enables more rewrites)
    //   2. optimize_for_inference: const-fold, fuse BatchNorm into Conv, dead code,
    //      remove dropout in eval, fuse linear chains, etc.
    // Both are one-time cost at model-load. Typical inference speedup 1.5-3×.
    // Set LAGAT_LC_NO_OPTIM=1 to skip (debug only).
    if (std::getenv("LAGAT_LC_NO_OPTIM") == nullptr) {
        try {
            module_ = torch::jit::freeze(module_);
        } catch (const std::exception& e) {
            std::cerr << "[lc-mapf] torch::jit::freeze failed: " << e.what()
                      << " (continuing without freeze)" << std::endl;
        }
        try {
            module_ = torch::jit::optimize_for_inference(module_);
            std::cerr << "[lc-mapf] applied freeze + optimize_for_inference" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[lc-mapf] torch::jit::optimize_for_inference failed: "
                      << e.what() << " (continuing)" << std::endl;
        }
    }
}

void LCMAPFTorchscriptModel::check_batched_shapes(
    const torch::Tensor& observations,
    const torch::Tensor& agent_chat_ids
) {
    if (observations.dim() != 3) {
        throw std::invalid_argument("observations must be rank-3 [B, C, T]");
    }
    if (agent_chat_ids.dim() != 3) {
        throw std::invalid_argument("agent_chat_ids must be rank-3 [B, C, L]");
    }
    if (observations.size(0) != agent_chat_ids.size(0) || observations.size(1) != agent_chat_ids.size(1)) {
        throw std::invalid_argument("observations and agent_chat_ids must have matching [B, C]");
    }
}

torch::Tensor LCMAPFTorchscriptModel::action_probs(
    const torch::Tensor& observations,
    const torch::Tensor& agent_chat_ids
) {
    // Inference mode: disables autograd machinery (version counter, dispatch
    // tracking) — significant overhead for many small ops in NN forward.
    c10::InferenceMode guard(true);

    check_batched_shapes(observations, agent_chat_ids);

    // === AOTI path ===
    if (aoti_mode_) {
        auto obs = observations.to(device_, /*dtype=*/torch::kLong, /*non_blocking=*/false, /*copy=*/false);
        auto chat = agent_chat_ids.to(device_, /*dtype=*/torch::kLong, /*non_blocking=*/false, /*copy=*/false);
        const int64_t actual_n = obs.size(1);

        // Dynamic-N single-package mode: no dispatch, no padding.
        // Convention: aoti_loaders_ has exactly 1 entry with key=0.
        if (aoti_loaders_.size() == 1 && aoti_loaders_.front().first == 0) {
            auto outs = aoti_loaders_.front().second->run({obs, chat});
            if (outs.empty()) {
                throw std::runtime_error("AOTI dynamic: runner returned no outputs");
            }
            return outs[0];
        }

        // Fixed-N dispatch mode: find smallest pre-compiled N >= actual_n, pad.
        auto it = std::lower_bound(
            aoti_loaders_.begin(), aoti_loaders_.end(), actual_n,
            [](const auto& p, int64_t n) { return p.first < n; });
        if (it == aoti_loaders_.end()) {
            throw std::runtime_error(
                "AOTI: no .pt2 for N=" + std::to_string(actual_n)
                + " (max compiled N=" + std::to_string(aoti_loaders_.back().first) + ")");
        }
        const int64_t target_n = it->first;

        torch::Tensor obs_in = obs;
        torch::Tensor chat_in = chat;
        if (actual_n < target_n) {
            const int64_t pad = target_n - actual_n;
            auto pad_obs = torch::zeros({obs.size(0), pad, obs.size(2)},
                torch::TensorOptions().dtype(obs.dtype()).device(obs.device()));
            auto pad_chat = torch::zeros({chat.size(0), pad, chat.size(2)},
                torch::TensorOptions().dtype(chat.dtype()).device(chat.device()));
            obs_in = torch::cat({obs, pad_obs}, /*dim=*/1);
            chat_in = torch::cat({chat, pad_chat}, /*dim=*/1);
        }

        auto outs = it->second->run({obs_in, chat_in});
        if (outs.empty()) {
            throw std::runtime_error("AOTI: runner returned no outputs");
        }
        torch::Tensor result = outs[0];
        // Output is [target_n, 5]. Slice back to actual_n if padded.
        if (actual_n < target_n) {
            result = result.slice(/*dim=*/0, /*start=*/0, /*end=*/actual_n);
        }
        return result;
    }

    // Autocast: ops cast to half (BF16/FP16) on the fly. Restore on exit.
    const bool use_autocast = (autocast_mode_ != AutocastMode::None);
    if (use_autocast) {
        at::autocast::set_autocast_enabled(at::kCUDA, true);
        at::autocast::set_autocast_dtype(at::kCUDA,
            autocast_mode_ == AutocastMode::BF16 ? at::kBFloat16 : at::kHalf);
    }
    auto autocast_cleanup = [&]() {
        if (use_autocast) {
            at::autocast::set_autocast_enabled(at::kCUDA, false);
        }
    };

    // Diagnostic: log first 5 shapes to confirm stability for CUDA Graph eligibility.
    if (std::getenv("LAGAT_LC_SHAPE_LOG") != nullptr) {
        static int slog_calls = 0;
        if (slog_calls < 5) {
            std::cerr << "[lc-shape] call=" << slog_calls
                      << " obs=" << observations.size(0) << "x" << observations.size(1) << "x" << observations.size(2)
                      << " chat=" << agent_chat_ids.size(0) << "x" << agent_chat_ids.size(1) << "x" << agent_chat_ids.size(2)
                      << std::endl;
            ++slog_calls;
        }
    }

    static const bool deep_profile = std::getenv("LAGAT_LC_DEEP_PROFILE") != nullptr;
    static long long t_h2d_us = 0, t_dispatch_us = 0, t_compute_us = 0;
    static int dp_calls = 0;
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto us = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };

    auto t0 = now();
    auto obs = observations.to(device_, /*dtype=*/torch::kLong, /*non_blocking=*/false, /*copy=*/false);
    auto chat = agent_chat_ids.to(device_, /*dtype=*/torch::kLong, /*non_blocking=*/false, /*copy=*/false);
    if (deep_profile && device_.is_cuda()) {
#if LAGAT_LC_HAS_CUDA_HEADERS
        c10::cuda::CUDACachingAllocator::emptyCache();  // no-op for sync, kept for safety
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
#endif
    }
    auto t1 = now();

    torch::Tensor result;

    if (graph_enabled_) {
#if LAGAT_LC_HAS_CUDA_HEADERS
        // Lazy-init static buffers on first call.
        if (!static_obs_.defined()) {
            static_obs_ = torch::empty(obs.sizes(),
                torch::TensorOptions().dtype(torch::kLong).device(device_));
            static_chat_ = torch::empty(chat.sizes(),
                torch::TensorOptions().dtype(torch::kLong).device(device_));
        }
        // Verify shape stability (one-time check).
        const bool shape_match = static_obs_.sizes() == obs.sizes()
                              && static_chat_.sizes() == chat.sizes();
        if (!shape_match) {
            // Shape changed → fall back to regular forward (rare).
            std::cerr << "[lc-mapf] CUDA Graph: shape mismatch, fallback to regular forward"
                      << std::endl;
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(obs);
            inputs.emplace_back(chat);
            result = module_.forward(inputs).toTensor();
        } else if (warmup_remaining_ > 0) {
            // Warmup: regular forward into static buffers (so compiles, caches warm up).
            static_obs_.copy_(obs);
            static_chat_.copy_(chat);
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(static_obs_);
            inputs.emplace_back(static_chat_);
            result = module_.forward(inputs).toTensor();
            --warmup_remaining_;
        } else if (!graph_captured_) {
            // Capture phase: must run on non-default stream. Create dedicated
            // stream, switch, run forward inside CUDAGraph capture, restore.
            static_obs_.copy_(obs);
            static_chat_.copy_(chat);
            cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
            // Create non-default stream for capture; reuse for replays.
            auto capture_stream = c10::cuda::getStreamFromPool(/*high_priority=*/false, device_.index());
            c10::cuda::CUDAStreamGuard stream_guard(capture_stream);
            graph_ = std::make_unique<at::cuda::CUDAGraph>();
            graph_->capture_begin();
            std::vector<torch::jit::IValue> inputs;
            inputs.emplace_back(static_obs_);
            inputs.emplace_back(static_chat_);
            static_out_ = module_.forward(inputs).toTensor();
            graph_->capture_end();
            graph_captured_ = true;
            std::cerr << "[lc-mapf] CUDA Graph captured (out shape "
                      << static_out_.size(0) << "x" << static_out_.size(1) << ")"
                      << std::endl;
            // First post-capture run: graph already executed during capture.
            result = static_out_;
        } else {
            // Replay: copy inputs into static buffers, replay graph.
            static_obs_.copy_(obs);
            static_chat_.copy_(chat);
            graph_->replay();
            // static_out_ holds the new output.
            result = static_out_;
        }
#else
        throw std::runtime_error(
            "CUDA Graph execution requires CUDA development headers");
#endif
    } else {
        std::vector<torch::jit::IValue> inputs;
        inputs.emplace_back(obs);
        inputs.emplace_back(chat);
        result = module_.forward(inputs).toTensor();
    }
    auto t2 = now();
    if (deep_profile && device_.is_cuda()) {
#if LAGAT_LC_HAS_CUDA_HEADERS
        cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream());
#endif
    }
    auto t3 = now();

    if (deep_profile) {
        t_h2d_us     += us(t0, t1);
        t_dispatch_us+= us(t1, t2);
        t_compute_us += us(t2, t3);
        ++dp_calls;
        if (dp_calls % 50 == 0) {
            std::cerr << "[lc-deep-profile] calls=" << dp_calls
                      << " sums(ms): h2d_inputs+sync=" << t_h2d_us / 1000
                      << " dispatch_only=" << t_dispatch_us / 1000
                      << " gpu_compute_sync=" << t_compute_us / 1000
                      << std::endl;
        }
    }
    autocast_cleanup();
    return result;
}

torch::Tensor LCMAPFTorchscriptModel::vector2d_to_long_tensor(const std::vector<std::vector<int64_t>>& data) {
    if (data.empty()) {
        throw std::invalid_argument("Input 2D vector must not be empty");
    }
    const int64_t rows = static_cast<int64_t>(data.size());
    const int64_t cols = static_cast<int64_t>(data.front().size());
    if (cols == 0) {
        throw std::invalid_argument("Input 2D vector must have non-empty rows");
    }

    std::vector<int64_t> flat;
    flat.reserve(static_cast<size_t>(rows * cols));
    for (const auto& row : data) {
        if (static_cast<int64_t>(row.size()) != cols) {
            throw std::invalid_argument("Input 2D vector rows must all have equal length");
        }
        flat.insert(flat.end(), row.begin(), row.end());
    }

    return torch::from_blob(flat.data(), {rows, cols}, torch::TensorOptions().dtype(torch::kLong))
        .clone();
}

torch::Tensor LCMAPFTorchscriptModel::action_probs_from_vectors(
    const std::vector<std::vector<int64_t>>& observations,
    const std::vector<std::vector<int64_t>>& agent_chat_ids
) {
    auto obs = vector2d_to_long_tensor(observations).unsqueeze(0);   // [1, C, T]
    auto chat = vector2d_to_long_tensor(agent_chat_ids).unsqueeze(0);  // [1, C, L]
    return action_probs(obs, chat);
}

std::vector<std::vector<float>> LCMAPFTorchscriptModel::action_probs_from_vectors_std(
    const std::vector<std::vector<int64_t>>& observations,
    const std::vector<std::vector<int64_t>>& agent_chat_ids
) {
    // LAGAT_LC_PROFILE=1: split timing between vec→tensor build, GPU forward,
    // GPU→CPU copy, and vec convert. Helps target the right opt.
    static const bool lc_profile = std::getenv("LAGAT_LC_PROFILE") != nullptr;
    static long long t_build_us = 0, t_forward_us = 0, t_d2h_us = 0, t_pack_us = 0;
    static int call_idx = 0;
    auto now = []() { return std::chrono::steady_clock::now(); };
    auto us = [](auto a, auto b) {
        return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count();
    };

    auto t0 = now();
    auto obs = vector2d_to_long_tensor(observations).unsqueeze(0);
    auto chat = vector2d_to_long_tensor(agent_chat_ids).unsqueeze(0);
    auto t1 = now();
    auto probs_gpu = action_probs(obs, chat);
    auto t2 = now();
    auto probs = probs_gpu.to(torch::kCPU).contiguous();
    auto t3 = now();

    if (probs.dim() != 2) {
        throw std::runtime_error("Expected 2D probability tensor [N, 5]");
    }
    const int64_t rows = probs.size(0);
    const int64_t cols = probs.size(1);
    const float* data = probs.data_ptr<float>();
    std::vector<std::vector<float>> out(static_cast<size_t>(rows), std::vector<float>(static_cast<size_t>(cols)));
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            out[static_cast<size_t>(i)][static_cast<size_t>(j)] = data[i * cols + j];
        }
    }
    auto t4 = now();

    if (lc_profile) {
        t_build_us   += us(t0, t1);
        t_forward_us += us(t1, t2);
        t_d2h_us     += us(t2, t3);
        t_pack_us    += us(t3, t4);
        ++call_idx;
        if (call_idx % 50 == 0) {
            std::cerr << "[lc-fwd-profile] calls=" << call_idx
                      << " sums(ms): vec_to_tensor=" << t_build_us / 1000
                      << " forward=" << t_forward_us / 1000
                      << " d2h=" << t_d2h_us / 1000
                      << " pack_vec=" << t_pack_us / 1000
                      << std::endl;
        }
    }
    return out;
}

std::vector<int64_t> LCMAPFTorchscriptModel::greedy_actions(
    const torch::Tensor& observations,
    const torch::Tensor& agent_chat_ids
) {
    auto probs = action_probs(observations, agent_chat_ids);
    auto actions = std::get<1>(probs.max(-1, false)).to(torch::kCPU).contiguous();
    return std::vector<int64_t>(actions.data_ptr<int64_t>(), actions.data_ptr<int64_t>() + actions.numel());
}

std::vector<int64_t> LCMAPFTorchscriptModel::greedy_actions_from_vectors(
    const std::vector<std::vector<int64_t>>& observations,
    const std::vector<std::vector<int64_t>>& agent_chat_ids
) {
    auto probs = action_probs_from_vectors(observations, agent_chat_ids);
    auto actions = std::get<1>(probs.max(-1, false)).to(torch::kCPU).contiguous();
    return std::vector<int64_t>(actions.data_ptr<int64_t>(), actions.data_ptr<int64_t>() + actions.numel());
}

}  // namespace lc_mapf




// Note: this file is now a pure C++ implementation used by LaGAT; the original
// pybind11 bindings and cppimport configuration have been removed.

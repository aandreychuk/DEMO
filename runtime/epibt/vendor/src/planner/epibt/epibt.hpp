#pragma once

#include <environment/robot.hpp>
#include <planner/epibt/operations.hpp>
#include <utils/time.hpp>

// Enhanced Priority Inheritance with BackTracking
class EPIBT {
protected:
    Robots& robots;

    TimePoint end_time;

    double cur_score = 0;

    double old_score = 0;

    uint32_t available_operation_depth = -1;

    uint32_t visited_counter = 1;

    uint32_t recursion_counter = 0;

    uint32_t inheritance_priority = 0;

    std::vector<uint32_t> desires;

    std::vector<uint32_t> curr_visited;

    std::vector<uint32_t> visited;

    std::vector<uint32_t> visited_num;

    std::vector<uint32_t> order;

    std::vector<double> robot_power;

    // smart_dist_dp[r][desired] = get_smart_dist_IMPL(r, desired)
    std::vector<std::vector<int64_t>> smart_dist_dp;

    // robot_desires[r] = { desired }
    std::vector<std::vector<uint32_t>> robot_desires;

    // used_edge[edge][depth] = robot id
    std::vector<std::array<uint32_t, EPIBT_DEPTH_VALUE>> used_edge;

    // used_pos[pos][depth] = robot id
    std::vector<std::array<uint32_t, EPIBT_DEPTH_VALUE>> used_pos;

    [[nodiscard]] bool validate_path(uint32_t r, uint32_t desired) const;

    [[nodiscard]] uint32_t get_used(uint32_t r) const;

    [[nodiscard]] int64_t get_smart_dist_IMPL(uint32_t r, uint32_t desired) const;

    [[nodiscard]] int64_t get_smart_dist(uint32_t r, uint32_t desired) const;

    void update_score(uint32_t r, uint32_t desired, int sign);

    void add_path(uint32_t r);

    void remove_path(uint32_t r);

    enum class RetType {
        FAILED,
        ACCEPTED,// success + accepted
        REJECTED,// success + not accepted
    };

    RetType build_impl(uint32_t r);

    void build(uint32_t r);

public:
    EPIBT(Robots& robots, TimePoint end_time, const std::vector<uint32_t> &operations);

    void solve();

    [[nodiscard]] double get_score() const;

    [[nodiscard]] std::vector<uint32_t> get_desires() const;

    [[nodiscard]] std::vector<ActionType> get_actions() const;
};

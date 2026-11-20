#pragma once

#include <cstdint>
#include <string>

enum class MapType : uint8_t {
    RANDOM,
    CITY,
    GAME,
    SORTATION,
    WAREHOUSE,

    NUM
};

enum class PlannerType : uint8_t {
    PIBT,       // pibt
    CAUSAL_PIBT,// causal pibt
    PIBT_TF,    // pibt + trafficflow
    EPIBT,      // epibt
    EPIBT_LNS,  // epibt + lns
    PEPIBT_LNS, // parallel epibt + lns

    NUM
};

enum class SchedulerType : uint8_t {
    CONST,
    GREEDY,

    NUM
};

enum class GraphGuidanceType : uint8_t {
    DISABLE,
    ENABLE,

    NUM
};

// ==================================

std::string map_type_to_string(MapType type);

MapType string_to_map_type(const std::string &str);

std::string planner_type_to_string(PlannerType type);

PlannerType string_to_planner_type(const std::string &str);

std::string scheduler_type_to_string(SchedulerType type);

SchedulerType string_to_scheduler_type(const std::string &str);

std::string graph_guidance_type_to_string(GraphGuidanceType type);

GraphGuidanceType string_to_graph_guidance_type(const std::string &str);

// ==================================

MapType &get_map_type();

PlannerType &get_planner_type();

SchedulerType &get_scheduler_type();

GraphGuidanceType &get_graph_guidance_type();

uint32_t &get_step_time();

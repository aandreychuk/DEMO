#pragma once

#include <environment/action_model.hpp>

#include <cstdint>

class Position {
    uint16_t row = 0, col = 0;

#ifdef ENABLE_ROTATE_MODEL
    uint16_t dir = 0;
#endif

public:
    Position() = default;

#ifndef ENABLE_ROTATE_MODEL

    explicit
#endif
            Position(uint32_t pos
#ifdef ENABLE_ROTATE_MODEL
                     ,
                     uint32_t dir
#endif
            );

    Position(uint32_t x, uint32_t y
#ifdef ENABLE_ROTATE_MODEL
             ,
             uint32_t dir
#endif
    );

    [[nodiscard]] uint32_t get_pos() const;

    [[nodiscard]] uint32_t get_row() const;

    [[nodiscard]] uint32_t get_col() const;

#ifdef ENABLE_ROTATE_MODEL
    [[nodiscard]] uint32_t get_dir() const;
#endif

    [[nodiscard]] bool is_valid() const;

#ifdef ENABLE_ROTATE_MODEL
    [[nodiscard]] Position move_forward() const;

    [[nodiscard]] Position rotate() const;

    [[nodiscard]] Position counter_rotate() const;
#else

    [[nodiscard]] Position move_east() const;

    [[nodiscard]] Position move_south() const;

    [[nodiscard]] Position move_west() const;

    [[nodiscard]] Position move_north() const;

#endif

    [[nodiscard]] Position simulate(const ActionType &action) const;

    friend bool operator==(const Position &lhs, const Position &rhs);

    friend bool operator!=(const Position &lhs, const Position &rhs);

    friend bool operator<(const Position &lhs, const Position &rhs);
};

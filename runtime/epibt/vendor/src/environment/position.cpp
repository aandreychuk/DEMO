#include <environment/position.hpp>

#include <environment/map.hpp>
#include <utils/assert.hpp>

Position::Position(uint32_t pos
#ifdef ENABLE_ROTATE_MODEL
                   ,
                   uint32_t dir
#endif
                   )
    : row((pos - 1) / get_map().get_cols()),
      col((pos - 1) % get_map().get_cols())
#ifdef ENABLE_ROTATE_MODEL
      ,
      dir(dir)
#endif
{
    ASSERT(pos < get_map().get_size(), "invalid pos: " + std::to_string(pos));
    ASSERT(row < get_map().get_rows(), "invalid row: " + std::to_string(row));
    ASSERT(col < get_map().get_cols(), "invalid col: " + std::to_string(col));
#ifdef ENABLE_ROTATE_MODEL
    ASSERT(dir < DIRECTIONS_NUM, "invalid dir");
#endif
}

Position::Position(uint32_t row, uint32_t col
#ifdef ENABLE_ROTATE_MODEL
                   ,
                   uint32_t dir
#endif
                   )
    : row(row),
      col(col)
#ifdef ENABLE_ROTATE_MODEL
      ,
      dir(dir)
#endif
{
    ASSERT(row < get_map().get_rows(), "invalid row: " + std::to_string(row));
    ASSERT(col < get_map().get_cols(), "invalid col: " + std::to_string(col));
#ifdef ENABLE_ROTATE_MODEL
    ASSERT(dir < DIRECTIONS_NUM, "invalid dir");
#endif
}

uint32_t Position::get_pos() const {
    return row * get_map().get_cols() + col + 1;
}

uint32_t Position::get_row() const {
    return row;
}

uint32_t Position::get_col() const {
    return col;
}

#ifdef ENABLE_ROTATE_MODEL
uint32_t Position::get_dir() const {
    return dir;
}
#endif

bool Position::is_valid() const {
    return row < get_map().get_rows() &&
           col < get_map().get_cols() &&
           get_map().is_free(get_pos());
}

#ifdef ENABLE_ROTATE_MODEL
Position Position::move_forward() const {
    Position p = *this;
    ASSERT(dir < DIRECTIONS_NUM, "invalid position dir");
    if (dir == 0) {
        p.col++;
    } else if (dir == 1) {
        p.row++;
    } else if (dir == 2) {
        p.col--;
    } else if (dir == 3) {
        p.row--;
    }
    return p;
}

Position Position::rotate() const {
    Position p = *this;
    p.dir = (p.dir + 1) % DIRECTIONS_NUM;
    ASSERT(p.dir < DIRECTIONS_NUM, "invalid position dir");
    return p;
}

Position Position::counter_rotate() const {
    Position p = *this;
    p.dir = (p.dir + DIRECTIONS_NUM - 1) % DIRECTIONS_NUM;
    ASSERT(p.dir < DIRECTIONS_NUM, "invalid position dir");
    return p;
}
#else
Position Position::move_east() const {
    Position p = *this;
    p.col++;
    return p;
}

Position Position::move_south() const {
    Position p = *this;
    p.row++;
    return p;
}

Position Position::move_west() const {
    Position p = *this;
    p.col--;
    return p;
}

Position Position::move_north() const {
    Position p = *this;
    p.row--;
    return p;
}
#endif

Position Position::simulate(const ActionType &action) const {
#ifdef ENABLE_ROTATE_MODEL
    if (action == ActionType::FORWARD) {
        return move_forward();
    } else if (action == ActionType::ROTATE) {
        return rotate();
    } else if (action == ActionType::COUNTER_ROTATE) {
        return counter_rotate();
    }
#else
    if (action == ActionType::EAST) {
        return move_east();
    } else if (action == ActionType::SOUTH) {
        return move_south();
    } else if (action == ActionType::WEST) {
        return move_west();
    } else if (action == ActionType::NORTH) {
        return move_north();
    }
#endif
    ASSERT(action == ActionType::WAIT, "unexpected action");
    return *this;
}

bool operator==(const Position &lhs, const Position &rhs) {
    return lhs.row == rhs.row &&
           lhs.col == rhs.col
#ifdef ENABLE_ROTATE_MODEL
           && lhs.dir == rhs.dir
#endif
            ;
}

bool operator!=(const Position &lhs, const Position &rhs) {
    return !(lhs == rhs);
}

bool operator<(const Position &lhs, const Position &rhs) {
    if (lhs.row != rhs.row) {
        return lhs.row < rhs.row;
    }
    if (lhs.col != rhs.col) {
        return lhs.col < rhs.col;
    }
#ifdef ENABLE_ROTATE_MODEL
    return lhs.dir < rhs.dir;
#else
    return false;
#endif
}

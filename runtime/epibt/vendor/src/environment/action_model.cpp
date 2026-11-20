#include <environment/action_model.hpp>

#include <utils/assert.hpp>

#include <algorithm>

constexpr std::array<char, ACTIONS_NUM> ACTIONS_CHAR = {
#ifdef ENABLE_ROTATE_MODEL
        'F',
        'R',
        'C',
#else
        'E',
        'S',
        'W',
        'N',
#endif
        'w'
};

char action_to_char(ActionType action) {
    ASSERT(static_cast<uint32_t>(action) < ACTIONS_CHAR.size(), "invalid action");
    return ACTIONS_CHAR[static_cast<uint32_t>(action)];
}

ActionType char_to_action(char symbol) {
    auto it = std::find(ACTIONS_CHAR.begin(), ACTIONS_CHAR.end(), symbol);
    ASSERT(it != ACTIONS_CHAR.end(), "invalid char");
    return static_cast<ActionType>(it - ACTIONS_CHAR.begin());
}

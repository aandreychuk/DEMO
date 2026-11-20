#include <planner/epibt/operations.hpp>

#include <environment/position.hpp>
#include <utils/assert.hpp>

#include <algorithm>
#include <fstream>
#include <set>
#include <sstream>
#include <tuple>

class OperationsGenerator {
    std::vector<Operation> pool;

    void generate(Operation &op, uint32_t i);

public:
    std::vector<Operation> get();
};

bool verify_operation(const Operation &op) {
#ifdef ENABLE_ROTATE_MODEL
    // не нужно, чтобы на конце операции были повороты
    // грамотный подсчет веса операции это увидит
    for (int i = op.size() - 1; i >= 0 && op[i] != ActionType::FORWARD; i--) {
        if (op[i] == ActionType::ROTATE || op[i] == ActionType::COUNTER_ROTATE) {
            return false;
        }
    }
    // операции вида CWR плохие, потому что мы повернулись, прошло пашу шагов и мы повернулись обратно
    for (int i = 0; i < op.size(); i++) {
        for (int j = i + 1; j < op.size() && op[j] != ActionType::FORWARD; j++) {
            if (op[i] == ActionType::ROTATE && op[j] == ActionType::COUNTER_ROTATE) {
                return false;
            }
            if (op[i] == ActionType::COUNTER_ROTATE && op[j] == ActionType::ROTATE) {
                return false;
            }
        }
    }
    // операции вида CWC плохие, потому что мы повернулись, подождали и повернулись еще раз
    // FAILED_ASSERT("TODO");

    // операции вида RRR плохие, потому что мы 3 раза крутимся, когда могли 1: CWW
    for (int i = 0; i < op.size(); i++) {
        for (int j = i + 1; j < op.size() && op[j] != ActionType::FORWARD; j++) {
            for (int k = j + 1; k < op.size() && op[k] != ActionType::FORWARD; k++) {
                if ((op[i] == ActionType::ROTATE || op[i] == ActionType::COUNTER_ROTATE) && op[i] == op[j] &&
                    op[j] == op[k]) {
                    return false;
                }
            }
        }
    }
#endif
    return true;
}

void OperationsGenerator::generate(Operation &op, uint32_t i) {
    if (i == EPIBT_DEPTH_VALUE) {
        if (verify_operation(op)) {
            pool.push_back(op);
        }
    } else {
        for (int32_t action = 0; action < ACTIONS_NUM; action++) {
            op[i] = static_cast<ActionType>(action);
            generate(op, i + 1);
        }
    }
}

static inline std::vector<uint32_t> operation_weight;

std::vector<Operation> OperationsGenerator::get() {
    Operation op;
    generate(op, 0);

    // read pool
    /*{
        std::vector<std::string> operations_pool_strs = {
                "17 WWW RRF RWF CWF WWF RFW CFW WFW RFF CFF FWW WFF FRF FCF FWF FFW FFF",

                "48 WWWW RRWF RWWF CWWF WWWF RRFW RWFW CWFW WWFW RRFF RFWW RWFF CFWW CWFF WFWW WWFF RFRF RFCF CFRF CFCF RFWF CFWF WFRF WFCF FWWW WFWF FRRF RFFW CFFW FRWF FCWF FWWF WFFW RFFF CFFF FRFW FCFW FWFW WFFF FRFF FCFF FFWW FWFF FFRF FFCF FFWF FFFW FFFF",

                "136 WWWWW CCWWF RWWWF CWWWF WWWWF RRWFW RWWFW CWWFW WWWFW RRFWW RRWFF RWFWW RWWFF CWFWW CWWFF RRFRF RRFCF WWFWW WWWFF RRFWF RWFRF RWFCF CWFRF CWFCF RFWWW RWFWF CFWWW CWFWF WWFRF WWFCF WFWWW WWFWF RFRRF CFRRF RRFFW RFRWF RFCWF CFRWF CFCWF RFWWF RWFFW CFWWF CWFFW WFRRF WFRWF WFCWF FWWWW WFWWF WWFFW RRFFF RFRFW RFCFW CFRFW CFCFW FRRWF RFWFW RWFFF CFWFW CWFFF FRWWF FCWWF WFRFW WFCFW FWWWF WFWFW WWFFF RFRFF RFCFF CFRFF CFCFF FRRFW RFFWW RFWFF CFFWW CFWFF FRWFW FCWFW WFRFF WFCFF FWWFW WFFWW WFWFF FRRFF RFFRF RFFCF CFFRF CFFCF RFFWF CFFWF FRFWW FRWFF FCFWW FCWFF WFFRF WFFCF FWFWW FWWFF WFFWF FRFRF FRFCF FCFRF FCFCF RFFFW CFFFW FRFWF FCFWF FWFRF FWFCF FFWWW FWFWF WFFFW FFRRF RFFFF CFFFF FRFFW FCFFW FFRWF FFCWF FFWWF FWFFW WFFFF FRFFF FCFFF FFRFW FFCFW FFWFW FWFFF FFRFF FFCFF FFFWW FFWFF FFFRF FFFCF FFFWF FFFFW FFFFF",
        };
        std::stringstream input(operations_pool_strs.at(EPIBT_DEPTH_VALUE - 3));
        uint32_t num;
        input >> num;
        for (uint32_t i = 0; i < num; i++) {
            std::string str;
            input >> str;
            Operation op;
            str.resize(DEPTH, 'W');
            //ASSERT(str.size() == op.size(), "does not match sizes: >" + str + "<, " + std::to_string(op.size()));
            std::stringstream ss(str);
            ss >> op;
            pool.push_back(op);
        }
    }*/

    auto get_operation_weight = [&](Operation op) {
        int64_t score = 0;
        for (uint32_t d = 0; d < op.size(); d++) {
#ifdef ENABLE_ROTATE_MODEL
            score += (op[d] == ActionType::FORWARD) * static_cast<int64_t>(op.size() - d + 1) * 10;
            score -= (op[d] == ActionType::WAIT) * static_cast<int64_t>(op.size() - d) * 3;
            score -= (op[d] == ActionType::ROTATE) * static_cast<int64_t>(op.size() - d) * 2;
            score -= (op[d] == ActionType::COUNTER_ROTATE) * static_cast<int64_t>(op.size() - d) * 2;
#else
            /*score += (op[d] == ActionType::EAST) * static_cast<int64_t>(op.size() - d) * 10;
            score += (op[d] == ActionType::SOUTH) * static_cast<int64_t>(op.size() - d) * 10;
            score += (op[d] == ActionType::WEST) * static_cast<int64_t>(op.size() - d) * 10;
            score += (op[d] == ActionType::NORTH) * static_cast<int64_t>(op.size() - d) * 10;
            score -= (op[d] == ActionType::WAIT) * static_cast<int64_t>(op.size() - d) * 1;*/
#endif
        }

#ifndef ENABLE_ROTATE_MODEL
        std::set<std::pair<uint32_t, uint32_t>> visited;
        Position p;
        visited.insert({0, 0});
        for (uint32_t d = 0; d < op.size(); d++) {
            if (op[d] == ActionType::WAIT) {
                score -= 10000 * (op.size() - d);
            }
        }
        for (uint32_t d = 0; d < op.size(); d++) {
            p = p.simulate(op[d]);
            if (visited.contains({p.get_row(), p.get_col()})) {
                score -= 5 * (op.size() - d + 1);
            }
            visited.insert({p.get_row(), p.get_col()});
        }
        for (uint32_t d = 0; d + 1 < op.size(); d++) {
            if (op[d] == op[d + 1]) {
                score += 1;
            }
        }
#endif
        return score;
    };
    std::stable_sort(pool.begin(), pool.end(), [&](auto lhs, auto rhs) {
        return get_operation_weight(lhs) < get_operation_weight(rhs);
    });

    // add WWW
    {
        Operation op;
        for (uint32_t i = 0; i < op.size(); i++) {
            op[i] = ActionType::WAIT;
        }
        auto it = std::find(pool.begin(), pool.end(), op);
        if (it != pool.end()) {
            pool.erase(it);
        }
        pool.insert(pool.begin(), op);
    }

    std::vector<Operation> result;

    std::set<std::tuple<uint32_t, std::array<std::pair<uint32_t, uint32_t>, EPIBT_DEPTH_VALUE>>> visited;
    for (auto operation: pool) {
        std::array<std::pair<uint32_t, uint32_t>, EPIBT_DEPTH_VALUE> positions{};
        Position p;
        std::set<std::pair<uint32_t, uint32_t>> visited_poses;
        visited_poses.insert({p.get_row(), p.get_col()});
        for (uint32_t d = 0; d < EPIBT_DEPTH_VALUE; d++) {
            p = p.simulate(operation[d]);
            positions[d] = {p.get_row(), p.get_col()};
            visited_poses.insert(positions[d]);
        }
        std::tuple<uint32_t, std::array<std::pair<uint32_t, uint32_t>, EPIBT_DEPTH_VALUE>> kek = {
#ifdef ENABLE_ROTATE_MODEL
                p.get_dir()
#else
                0
#endif
                        ,
                positions};
        if (!visited.count(kek)) {
            visited.insert(kek);
            result.push_back(operation);
        }
    }

    {
        int64_t prev_operation_weight = -1;
        uint32_t new_weight = -1;
        for (auto operation: result) {
            if (prev_operation_weight != get_operation_weight(operation)) {
                prev_operation_weight = get_operation_weight(operation);
                new_weight++;
            }
            operation_weight.push_back(new_weight);
        }
    }

    std::cout << "Operations:\n"
              << result.size() << '\n';
    for (uint32_t i = 0; i < result.size(); i++) {
        auto operation = result[i];
        std::cout << operation << ' ';// << get_operation_weight(operation) << ' ' << operation_weight[i] << '\n';
    }
    std::cout << std::endl;
    //std::exit(100);
    return result;
}

std::vector<Operation> &get_operations() {
    static std::vector<Operation> operations;
    return operations;
}

static inline std::vector<uint32_t> operation_depth;

uint32_t get_operation_depth(uint32_t index) {
    ASSERT(0 <= index && index < operation_depth.size(), "invalid index");
    ASSERT(operation_depth.size() == get_operations().size(), "unmatch sizes");
    return operation_depth[index];
}

static inline std::vector<uint32_t> operation_next;

uint32_t get_operation_next(uint32_t index) {
    ASSERT(0 <= index && index < operation_next.size(), "invalid index");
    ASSERT(operation_next.size() == get_operations().size(), "unmatch sizes");
    return operation_next[index];
}

std::vector<uint32_t> &get_operations_ids(uint32_t d) {
    static std::array<std::vector<uint32_t>, EPIBT_DEPTH_VALUE + 1> data;
    ASSERT(0 <= d && d <= EPIBT_DEPTH_VALUE, "invalid d");
    return data[d];
}

uint32_t get_operation_weight(uint32_t index) {
    ASSERT(0 <= index && index < operation_weight.size(), "invalid index");
    ASSERT(operation_weight.size() == get_operations().size(), "unmatch sizes");
    return operation_weight[index];
}

void init_operations() {
    get_operations() = OperationsGenerator().get();

    auto get_operation_depth = [&](const Operation &op) {
        uint32_t d = op.size();
        for (; d > 0 && op[d - 1] == ActionType::WAIT; d--) {
        }
        return d;
    };

    auto get_operation_next = [&](Operation op) {
        // ABC -> BCW

        for (uint32_t i = 0; i + 1 < EPIBT_DEPTH_VALUE; i++) {
            op[i] = op[i + 1];
        }
        op.back() = ActionType::WAIT;
        for (uint32_t i = 0; i < get_operations().size(); i++) {
            if (get_operations()[i] == op) {
                return i;
            }
        }
        FAILED_ASSERT("failed to get next operation");
        return uint32_t();
    };

    operation_depth.resize(get_operations().size());
    operation_next.resize(get_operations().size());
    for (uint32_t i = 0; i < get_operations().size(); i++) {
        operation_next[i] = get_operation_next(get_operations()[i]);
        uint32_t d = get_operation_depth(get_operations()[i]);
        operation_depth[i] = d;
        get_operations_ids(d).push_back(i);
    }
}

std::ostream &operator<<(std::ostream &output, const Operation &op) {
    for (auto action: op) {
        output << action_to_char(action);
    }
    return output;
}

std::istream &operator>>(std::istream &input, Operation &op) {
    for (auto &action: op) {
        char symbol;
        input >> symbol;
        action = char_to_action(symbol);
    }
    return input;
}

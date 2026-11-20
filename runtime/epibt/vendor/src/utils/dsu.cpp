#include <utils/dsu.hpp>

#include <utils/assert.hpp>

#include <numeric>

DSU::DSU(uint32_t n) : parent(n), size(n, 1) {
    std::iota(parent.begin(), parent.end(), 0);
}

uint32_t DSU::get(uint32_t v) {
    ASSERT(v < parent.size(), "out of range");
    if (parent[v] == v) {
        return v;
    } else {
        return parent[v] = get(parent[v]);
    }
}

uint32_t DSU::get_size(uint32_t v) {
    return size[get(v)];
}

void DSU::uni(uint32_t a, uint32_t b) {
    a = get(a);
    b = get(b);
    if (a == b) {
        return;
    }

    if (size[a] > size[b]) {
        std::swap(a, b);
    }
    // size[a] <= size[b]
    // a -> b

    parent[a] = b;
    size[b] += size[a];
}

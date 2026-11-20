#pragma once

#include <string>
#ifndef __EMSCRIPTEN__
#include <thread>
#endif
#include <vector>

std::vector<std::string> split(const std::string &str, char separator);

template<typename foo_t>
void launch_threads(const uint32_t threads_num, foo_t &&foo) {
#ifdef __EMSCRIPTEN__
    for (uint32_t thr = 0; thr < threads_num; ++thr) foo(thr);
#else
    std::vector<std::thread> threads(threads_num);
    for (uint32_t thr = 0; thr < threads.size(); thr++) {
        threads[thr] = std::thread(foo, thr);
    }
    for (uint32_t thr = 0; thr < threads.size(); thr++) {
        threads[thr].join();
    }
#endif
}

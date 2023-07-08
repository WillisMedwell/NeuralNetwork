#pragma once

#include <chrono>
#include <cstdint>
#include <limits>
#include <random>
#include <iostream>

namespace Tools {
float randomNum(float min, float max)
{
    thread_local static auto rd = std::random_device {};
    thread_local static auto gen = std::mt19937 { rd() };
    thread_local static auto dist = std::uniform_real_distribution { min, max };
    return dist(gen);
}

struct Timerbomb {
    std::chrono::steady_clock::time_point start;

    Timerbomb()
        : start(std::chrono::high_resolution_clock::now())
    {
    }
    ~Timerbomb()
    {
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
        std::cout << "\n" << duration << "ms" << std::endl;
    }
};
} // namespace Tools
#include "Math.hpp"
#include "Neural.hpp"

#include <array>
#include <chrono>
#include <execution>
#include <format>
#include <functional>
#include <tuple>
#include <vector>

int main()
{
    auto start = std::chrono::high_resolution_clock::now();

    // xor behaviour
    auto td = std::vector<NeuralTrainingSample> {
        { { 0, 0 }, { 0 } },
        { { 0, 1 }, { 1 } },
        { { 1, 0 }, { 1 } },
        { { 1, 1 }, { 0 } },
    };

    // [ 2, 1 ] network layout
    NeuralNetwork network {
        NeuralLayer<2, 2, Math::Activation::Gaussian>(),
        NeuralLayer<2, 1, Math::Activation::Sigmoid>()
    };

    // returns the best random network which is slightly trained.
    network.trainManyRandomNetworks<10000>(td, 100, 0.01f, 0.01f);
    std::cout << "Perf rand: " << network.performance(td) << '\n';

    // further train the best one.
    network.train(td, 2000000, 0.01f, 0.00001f);
    std::cout << "Perf after: " << network.performance(td) << "\n\n";

    std::cout << "0 xor 0 = " << network.propagateForward(std::vector<float> { 0, 0 }).front() << '\n';
    std::cout << "0 xor 1 = " << network.propagateForward(std::vector<float> { 0, 1 }).front() << '\n';
    std::cout << "1 xor 0 = " << network.propagateForward(std::vector<float> { 1, 0 }).front() << '\n';
    std::cout << "1 xor 1 = " << network.propagateForward(std::vector<float> { 1, 1 }).front() << "\n\n";
    std::cout << network;

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - start).count();
    std::cout << duration << "ms" << std::endl;
}

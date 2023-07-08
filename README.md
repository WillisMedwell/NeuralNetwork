# NeuralNetwork
An implementation of a basic neural network in Modern C++23. A learning exercise into neural networks, algorithms, and metaprogramming.  

## Features
- Gradient Descent
- Common Activation functions
- Networks stored in alignment

## Basic Usage
```cpp
NeuralNetwork network {
    NeuralLayer<2, 2, Math::Activation::Gaussian>(),
    NeuralLayer<2, 1, Math::Activation::Sigmoid>()
};

auto training_data = std::vector<NeuralTrainingSample> {
    { { 0, 0 }, { 0 } },
    { { 0, 1 }, { 1 } },
    { { 1, 0 }, { 1 } },
    { { 1, 1 }, { 0 } },
};

// [ 2, 1 ] network layout
NeuralNetwork network {
    NeuralLayer<2, 2, Math::Activation::Gaussian>(),
    NeuralLayer<2, 1, Math::Activation::Gaussian>()
};

// returns the best random network which is slightly trained.
network.trainManyRandomNetworks<10000>(training_data, 100, 0.01f, 0.01f);
std::cout << "Perf rand: " << network.performance(td) << '\n';

// further train the best one.
network.train(td, 2000000, 0.01f, 0.00001f);
std::cout << "Perf after: " << network.performance(td) << "\n";

std::cout << network << '\n';

```

## Typcial Output
The network is printed in a way where it can be copied directly back into code.  
```terminal
Perf rand: 0.0619224
Perf after: 1.89009e-05

0 xor 0 = 0.00505022
0 xor 1 = 0.996171
1 xor 0 = 0.994717
1 xor 1 = 0.00274341

NeuralNetwork network = {
        NeuralLayer<2, 2, Math::Activation::Gaussian>({{ {1.4414f, -1.3694f}, -0.0128874f }, { {0.408861f, 0.0375307f}, -0.037101f }}),
        NeuralLayer<2, 1, Math::Activation::Sigmoid>({{ {-12.7237f, 4.28367f}, 3.16051f }}),
};
```
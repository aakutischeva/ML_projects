#ifndef NEURAL_NET_HPP
#define NEURAL_NET_HPP

#include <vector>
#include <iostream>
#include <cmath>


class NeuralNetwork {
public:
    NeuralNetwork(const std::vector<int>& layerSizes);
    std::vector<double> feedforward(const std::vector<double>& inputs);
    void backpropagate(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate);
    std::vector<double> train(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& targets,
                         const std::vector<std::vector<double>>& check_data, const std::vector<std::vector<double>>& check_targets,
                         int epochs, double learningRate, double needed_accuracy);

    double print_train_result(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& targets);

private:
    std::vector<int> layer_sizes;
    std::vector<std::vector<std::vector<double>>> weights;
    std::vector<std::vector<double>> biases;
    std::vector<std::vector<double>> layers;

    std::vector<std::vector<std::vector<double>>> best_weights;

    void initializeWeights();
};

#endif // NEURAL_NET_HPP
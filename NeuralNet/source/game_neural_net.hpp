#include "neural_net.hpp"

NeuralNetwork &&game_neural_network(std::vector<double> (*generate_target)(const std::vector<double>&), std::vector<std::vector<double>> &data, std::vector<std::vector<double>> &targets, int overlearn_coefficient = 5,
                                    int epochs = 10000, double learningRate = 0.2, std::vector<int> layer_sizes = {36, 24, 16, 8},
                                    double needed_accuracy = 0.99);
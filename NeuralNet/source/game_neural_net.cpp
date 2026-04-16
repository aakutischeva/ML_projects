#include "game_neural_net.hpp"


NeuralNetwork &&game_neural_network(std::vector<double> (*generate_target)(const std::vector<double>&),
                                    std::vector<std::vector<double>> &data, std::vector<std::vector<double>> &targets,
                                    int overlearn_coefficient, int epochs, double learningRate,
                                    std::vector<int> layer_sizes, double needed_accuracy) {
    
    NeuralNetwork *nn = new NeuralNetwork(layer_sizes);

    std::vector<std::vector<double>> train_data, train_targets;

    for (int i = 0; i < (1 << 18); ++i) {

        std::vector<double> input(36, 0);

        // First 18 parameters divided into triplets
        for (int j = 0; j < 18; j += 3) {
            int triplet = (i >> j) & 0b111;
            if ((triplet & 0b001) == 0) {
                input[j] = 0;
                input[j+1] = 0;
                input[j+2] = 0;
            } else {
                input[j] = 1;
                input[j+1] = (triplet & 0b010) >> 1;
                input[j+2] = (triplet & 0b100) >> 2;
            }
        }

        // Next 4 parameters divided into duets
        for (int j = 18; j < 22; j += 2) {
            int duet = (i >> (j-18)) & 0b11;
            if ((duet & 0b01) == 0) {
                input[j] = 0;
                input[j+1] = 0;
            } else {
                input[j] = 1;
                input[j+1] = (duet & 0b10) >> 1;
            }
        }

        // Remaining 14 parameters divided into duets
        for (int j = 22; j < 36; j += 2) {
            int duet = (i >> (j-18)) & 0b11;
            if ((duet & 0b01) == 0) {
                input[j] = 0;
                input[j+1] = 0;
            } else {
                input[j] = 1;
                input[j+1] = (duet & 0b10) >> 1;
            }
        }

        data.push_back(input);
        if (i % overlearn_coefficient == 0) {
            train_data.push_back(input);
        }
    }


    for (int i = 0; i < data.size(); ++i) {
        targets.push_back(generate_target(data[i]));
    }


    for (int i = 0; i < train_data.size(); ++i) {
        train_targets.push_back(generate_target(train_data[i]));
    }

    std::cout << "Training data size: " << train_data.size() << " targets size: " << train_targets.size() << std::endl;
    std::cout << "Data size: " << data.size() << " targets size: " << targets.size() << std::endl;

    nn->train(train_data, train_targets, data, targets, epochs, learningRate, needed_accuracy);

    return std::move(*nn);
}

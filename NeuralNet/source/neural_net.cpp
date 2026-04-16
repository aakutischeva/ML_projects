#include "neural_net.hpp"



double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoidDerivative(double x) {
    return x * (1 - x);
}

NeuralNetwork::NeuralNetwork(const std::vector<int>& layerSizes) : layer_sizes(layerSizes) {
    initializeWeights();
}

void NeuralNetwork::initializeWeights() {
    srand(time(0));
    int num_layers = layer_sizes.size();

    weights.resize(num_layers - 1);
    biases.resize(num_layers - 1);
    layers.resize(num_layers);

    for (int l = 0; l < num_layers - 1; ++l) {
        weights[l].resize(layer_sizes[l], std::vector<double>(layer_sizes[l + 1]));
        biases[l].resize(layer_sizes[l + 1]);

        for (int i = 0; i < layer_sizes[l]; ++i) {
            for (int j = 0; j < layer_sizes[l + 1]; ++j) {
                weights[l][i][j] = ((double)rand() / RAND_MAX);
            }
        }

        for (int j = 0; j < layer_sizes[l + 1]; ++j) {
            biases[l][j] = ((double)rand() / RAND_MAX);
        }
    }
}

std::vector<double> NeuralNetwork::feedforward(const std::vector<double>& inputs) {
    layers[0] = inputs;

    for (int l = 0; l < layer_sizes.size() - 1; ++l) {
        layers[l + 1].resize(layer_sizes[l + 1]);
        for (int j = 0; j < layer_sizes[l + 1]; ++j) {
            layers[l + 1][j] = biases[l][j];
            for (int i = 0; i < layer_sizes[l]; ++i) {
                layers[l + 1][j] += layers[l][i] * weights[l][i][j];
            }
            layers[l + 1][j] = sigmoid(layers[l + 1][j]);
        }
    }
    return layers.back();
}

void NeuralNetwork::backpropagate(const std::vector<double>& inputs, const std::vector<double>& targets, double learningRate) {
    std::vector<double> outputErrors = targets;
    for (int i = 0; i < targets.size(); ++i) {
        outputErrors[i] -= layers.back()[i];
    }

    std::vector<std::vector<double>> deltas(layer_sizes.size());
    deltas.back() = outputErrors;

    for (int l = layer_sizes.size() - 2; l >= 0; --l) {
        deltas[l].resize(layer_sizes[l]);
        for (int i = 0; i < layer_sizes[l]; ++i) {
            double error = 0.0;
            for (int j = 0; j < layer_sizes[l + 1]; ++j) {
                error += deltas[l + 1][j] * weights[l][i][j];
            }
            deltas[l][i] = error * sigmoidDerivative(layers[l][i]);
        }

        for (int i = 0; i < layer_sizes[l]; ++i) {
            for (int j = 0; j < layer_sizes[l + 1]; ++j) {
                weights[l][i][j] += learningRate * deltas[l + 1][j] * layers[l][i];
            }
        }

        for (int j = 0; j < layer_sizes[l + 1]; ++j) {
            biases[l][j] += learningRate * deltas[l + 1][j];
        }
    }
}

std::vector<double> NeuralNetwork::train(const std::vector<std::vector<double>>& train_data, const std::vector<std::vector<double>>& train_targets,
                                    const std::vector<std::vector<double>>& check_data, const std::vector<std::vector<double>>& check_targets,
                                    int epochs, double learningRate, double needed_accuracy) {

    double min_error = MAXFLOAT;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        double total_error = 0;
        for (size_t i = 0; i < train_data.size(); ++i) {
            std::vector<double> outputs = feedforward(train_data[i]);
            backpropagate(train_data[i], train_targets[i], learningRate);
            for (int j = 0; j < train_targets[i].size(); ++j) {
                total_error += 0.5 * pow(train_targets[i][j] - outputs[j], 2);
            }
        }

        if (epoch % 10 == 9) {
            std::cout << "Epoch " << epoch + 1 << "/" << epochs << ", Error: " << total_error << std::endl;
        }

        if (total_error < min_error) {
            min_error = total_error;
            best_weights = weights;
        }

        if (epoch % 100 == 99) {

            std::cout << "=============================" << std::endl;
            print_train_result(train_data, train_targets);
            double accuracy = print_train_result(check_data, check_targets);
            std::cout << "Accuracy: " << accuracy << std::endl;
            std::cout << "=============================" << std::endl;
            if (accuracy > needed_accuracy) {
                break;
            }
        }
    }

    weights = best_weights;
    return layers.back();
}


double NeuralNetwork::print_train_result(const std::vector<std::vector<double>>& data, const std::vector<std::vector<double>>& targets) {

    size_t count = 0;

    for (int i = 0; i < data.size(); ++i) {
        std::vector<double> output = feedforward(data[i]);
        for (int j = 0; j < output.size(); ++j) {
            if (std::abs(output[j] - targets[i][j]) > 0.05) {
                ++count;
                break;
            }
        }
    }

    std::cout << "(Errors / Overall data): (" << count << " / " << data.size() << ")" << std::endl;

    return (data.size() - count) / double(data.size());

}

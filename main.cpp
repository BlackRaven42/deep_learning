#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <algorithm>
#include "mnist_reader.hpp"

#define NUM_OF_CLASSES     10
#define SIZE               784
#define LEARN_PAR          0.01
#define TRAIN_IMAGES       60000
#define TEST_IMAGES        10000
#define HIDDEN_UNITS       300
#define EPOCHS             10

void Calculate(const double* input, double* stealth, int S, double* output, double** w_ij, double** w_jk, double* bias_j, double* bias_k);
void BackPropagation(const double* input, double* stealth, int S, double n, double** w_ij, double** w_jk, double* bias_j, double* bias_k, double* out, double* o_err, double* s_err, uint8_t answer);

int main(int argc, char* argv[]) {
    int i, j, k;
    double n = LEARN_PAR;
    int S = HIDDEN_UNITS;
    int epochs = EPOCHS;
    char* path = "data";

    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-h")) {
            std::cout << "Available options:" << std::endl;
            std::cout << "-hu - number of hidden layer's neurons (int)" << std::endl;
            std::cout << "-n  - learning acceleration parameter (double)" << std::endl;
            std::cout << "-p  - path to the train & test samples (string)" << std::endl;
            std::cout << "-e - number of epochs (int)" << std::endl;
            return 0;
        }
        if ((!strcmp(argv[i], "-hu")) && (i + 1 < argc)) {
            S = atoi(argv[i + 1]);
            ++i;
        }
        else if ((!strcmp(argv[i], "-n")) && (i + 1 < argc)) {
            n = atof(argv[i + 1]);
            ++i;
        }
        else if ((!strcmp(argv[i], "-p")) && (i + 1 < argc)) {
            path = argv[i + 1];
            ++i;
        }
        else if ((!strcmp(argv[i], "-e")) && (i + 1 < argc)) {
            epochs = atoi(argv[i + 1]);
            ++i;
        }
    }

    mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(path);
    
    std::vector<std::vector<uint8_t>> train_images = dataset.training_images;
    std::vector<uint8_t> train_labels = dataset.training_labels;
    if (train_images.size() != train_labels.size()) {
        std::cout << "Error! Incorrect train set: number of images = " << train_images.size() << ", number of labels = " << train_labels.size();
        return 1;
    }
    int train_set_num = std::min(TRAIN_IMAGES, (int)train_images.size());

    double** normalized_train_images = new double*[train_set_num];
    for (i = 0; i < train_set_num; ++i) {
        normalized_train_images[i] = new double[SIZE]();
        for (j = 0; j < SIZE; ++j)
            normalized_train_images[i][j] = train_images[i][j] / 255.0;
    }

    std::vector<std::vector<uint8_t>> test_images = dataset.test_images;
    std::vector<uint8_t> test_labels = dataset.test_labels;
    if (test_images.size() != test_labels.size()) {
        std::cout << "Error! Incorrect test set: number of images = " << test_images.size() << ", number of labels = " << test_labels.size();
        return 2;
    }
    int test_set_num = std::min(TEST_IMAGES, (int)test_images.size());

    double** normalized_test_images = new double*[test_set_num];
    for (i = 0; i < test_set_num; ++i) {
        normalized_test_images[i] = new double[SIZE]();
        for (j = 0; j < SIZE; ++j)
            normalized_test_images[i][j] = test_images[i][j] / 255.0;
    }


    double** w_ij;
    w_ij = new double*[SIZE];
    for (i = 0; i < SIZE; ++i) {
        w_ij[i] = new double[S]();
    }
    double** w_jk;
    w_jk = new double*[S];
    for (j = 0; j < S; ++j) {
        w_jk[j] = new double[NUM_OF_CLASSES]();
    }

    double* bias_j;
    bias_j = new double[S]();
    double* bias_k;
    bias_k = new double[NUM_OF_CLASSES]();
    double* stealth;
    stealth = new double[S]();
    double* out;
    out = new double[NUM_OF_CLASSES]();
    double* o_err;
    o_err = new double[NUM_OF_CLASSES]();
    double* s_err;
    s_err = new double[S]();
    double hits_num = 0.0;

    // initializing of weights and biases with small different-sign values
    std::srand((unsigned int)time(0));
    for (j = 0; j < S; ++j) {
        for (i = 0; i < SIZE; ++i)
            w_ij[i][j] = (std::rand() % 1000 - 500) * 0.001;
        for (k = 0; k < NUM_OF_CLASSES; ++k)
            w_jk[j][k] = (std::rand() % 1000 - 500) * 0.001;
    }
    for (j = 0; j < S; ++j)
        bias_j[j] = (std::rand() % 1000 - 500) * 0.001;
    for (k = 0; k < NUM_OF_CLASSES; ++k)
        bias_k[k] = (std::rand() % 1000 - 500) * 0.001;

    
    // neural network training
    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (i = 0; i < train_set_num; i++) {
            int idx1 = rand() % train_set_num;
            int idx2 = rand() % train_set_num;
            std::swap(normalized_train_images[idx1], normalized_train_images[idx2]);
            std::swap(train_labels[idx1], train_labels[idx2]);
        }
        for (i = 0; i < test_set_num; i++) {
            int idx1 = rand() % test_set_num;
            int idx2 = rand() % test_set_num;
            std::swap(normalized_test_images[idx1], normalized_test_images[idx2]);
            std::swap(test_labels[idx1], test_labels[idx2]);
        }

        std::cout << std::endl << "Epoch " << epoch + 1 << "/" << epochs << ":\nTraining...";
        for (i = 0; i < train_set_num; ++i)
            BackPropagation(normalized_train_images[i], stealth, S, n, w_ij, w_jk, bias_j, bias_k, out, o_err, s_err, train_labels[i]);

        std::cout << std::endl << "Checking...";
        // neural network checking
        hits_num = 0.0;
        for (i = 0; i < train_set_num; ++i) {
            Calculate(normalized_train_images[i], stealth, S, out, w_ij, w_jk, bias_j, bias_k);
            hits_num += (std::distance(out, std::max_element(out, out + NUM_OF_CLASSES - 1))) == train_labels[i] ? 1.0 : 0.0;
        }
        std::cout << std::endl << "Results: " << std::endl;
        std::cout << "Train Accuracy = " << hits_num / train_set_num << std::endl;

        hits_num = 0.0;
        for (i = 0; i < test_set_num; ++i) {
            Calculate(normalized_test_images[i], stealth, S, out, w_ij, w_jk, bias_j, bias_k);
            hits_num += (std::distance(out, std::max_element(out, out + NUM_OF_CLASSES - 1))) == test_labels[i] ? 1.0 : 0.0;
        }
        std::cout << "Test Accuracy = " << hits_num / test_set_num << std::endl << std::endl;
    }

    system("pause");
    return 0;
}

void BackPropagation(const double* input, double* stealth, int S, double n, double** w_ij, double** w_jk, double* bias_j, double* bias_k, double* out, double* o_err, double* s_err, uint8_t answer) {
    int i = 0, j = 0, k = 0;    

    // forward step
    Calculate(input, stealth, S, out, w_ij, w_jk, bias_j, bias_k);

    // backward step
    // cross-entropy (error of output) & updating weights for output layer
    for (k = 0; k < NUM_OF_CLASSES; ++k) {
        o_err[k] = -out[k] + (answer == k ? 1.0 : 0.0);
        bias_k[k] += n * o_err[k];
        for (j = 0; j < S; ++j)
            w_jk[j][k] += n * stealth[j] * o_err[k];
    }

    // recalculating error & updating weights for hidden layer
    for (j = 0; j < S; ++j) {
        s_err[j] = 0.0;
        for (k = 0; k < NUM_OF_CLASSES; ++k)
            s_err[j] += o_err[k] * w_jk[j][k];
        s_err[j] *= stealth[j] * (1 - stealth[j]);

        bias_j[j] += n * s_err[j];
        for (i = 0; i < SIZE; ++i)
            w_ij[i][j] += n * input[i] * s_err[j];
    }
}

void Calculate(const double* input, double* stealth, int S, double* output, double** w_ij, double** w_jk, double* bias_j, double* bias_k) {
    int i = 0, j = 0, k = 0;
    double sum = 0.0;

    // calculating of hidden layer (logistic func)
    for (j = 0; j < S; ++j) {
        stealth[j] = bias_j[j];
        for (i = 0; i < SIZE; ++i)
            stealth[j] += input[i] * w_ij[i][j];
        stealth[j] = 1.0 / (1.0 + std::exp(-stealth[j]));
    }

    // calculating of output layer (soft-max func)
    for (k = 0; k < NUM_OF_CLASSES; ++k) {
        output[k] = bias_k[k];
        for (j = 0; j < S; ++j)
            output[k] += stealth[j] * w_jk[j][k];
        output[k] = std::exp(output[k]);
        sum += output[k];
    }
    for (k = 0; k < NUM_OF_CLASSES; ++k)
        output[k] /= sum;
}
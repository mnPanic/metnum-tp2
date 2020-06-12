//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include <string>
#include <fstream>

#include "pca.h"
#include "eigen.h"
#include "knn.h"

struct arguments {
    int method;
    std::string train_set;
    std::string test_set;
    std::string classif;
};

// Parsea los argumentos exitieando si no son del formato correcto.
arguments parse_argv(int argc, char** argv) {
    const int ARG_COUNT = 9;
    // $ ./tp2 -m <method> -i <train_set> -q <test_set> -o <classif>
    if (argc != ARG_COUNT) {
        std::cout
            << "Wrong number of args. Usage:\n\t"
            << "$ ./tp2 -m <method> -i <train_set> -q <test_set> -o <classif>"
            << std::endl;
        exit(1);
    }

    arguments args;

    for(int i = 1; i < ARG_COUNT; i+=2) {
        // i   es el arg
        // i+1 el val
        std::string arg = argv[i];
        std::string val = argv[i+1];

        // Como puede ser que c++ no te deje hacer switch en string?
        if(arg == "-m") args.method = atoi(val.c_str());
        if(arg == "-i") args.train_set = val;
        if(arg == "-q") args.test_set = val;
        if(arg == "-o") args.classif = val;
    }

    return args;
}

void vec2kagglecsv(Vector v, const std::string& path) {
    std::cout << "writing vector to csv file: " << path << std::endl;
    std::ofstream file(path);
    file << "ImageId,Label" << std::endl;
    for(int i = 0; i < v.rows(); i++) {
        file << i + 1 << "," << v.row(i) << std::endl;
    }

    std::cout << "done!" << std::endl;
}

Matrix csv2mat(const std::string& path) {
    std::cout << "reading matrix from file: " << path << std::endl;
    std::ifstream indata;
    indata.open(path);

    std::string line;
    std::vector<double> values;
    uint rows = 0;
    // ignore the first line
    std::getline(indata, line);
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;

        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }

        ++rows;
    }

    return Eigen::Map<Matrix>(values.data(), rows, values.size()/rows);
}

std::tuple<Matrix, Matrix> csv2matlabel(const std::string& path) {
    std::cout << "reading matrix & labels from file: " << path << std::endl;
    std::ifstream indata;
    indata.open(path);

    std::string line;
    std::vector<double> values;
    std::vector<double> labels;

    uint rows = 0;

    // ignore the first line
    std::getline(indata, line);

    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;

        // The first element is its label
        std::getline(lineStream, cell, ',');
        labels.push_back(std::stod(cell));

        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }

        ++rows;
    }

    return std::tuple<Matrix, Matrix>(
        Eigen::Map<Matrix>(values.data(), rows, values.size()/rows),
        Eigen::Map<Matrix>(labels.data(), rows, 1)
    );
}

constexpr int METHOD_KNN = 0;
constexpr int METHOD_KNN_PCA = 1;

int main(int argc, char** argv){
    arguments args = parse_argv(argc, argv);

    std::tuple<Matrix, Matrix> trainpair = csv2matlabel(args.train_set);
    Matrix train = std::get<0>(trainpair);
    Matrix labels = std::get<1>(trainpair);
    Matrix test = csv2mat(args.test_set);

    Vector pred;
    switch(args.method) {
        case METHOD_KNN:
        {
            std::cout << "Running kNN with k=4 and distance weights" << std::endl;

            KNNClassifier clf(4, "distance_pow");
            clf.fit(train, labels);
            pred = clf.predict(test);
            break;
        }
        case METHOD_KNN_PCA:
        {
            std::cout << "Running kNN+PCA with k=6 distance weights & alpha=34" << std::endl;
            KNNClassifier clf(6, "distance_pow_5");
            PCA pca(34);

            pca.fit(train);
            clf.fit(pca.transform(train), labels);
            pred = clf.predict(pca.transform(test));
            break;
        }
        default:
            std::cout << "Invalid method. 0: kNN y 1: kNN+PCA." << std::endl;
            exit(1);
    }

    vec2kagglecsv(pred, args.classif);
    return 0;
}

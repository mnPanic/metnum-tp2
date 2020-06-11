#include <algorithm>
//#include <chrono>
#include <iostream>
#include <math.h>
#include "knn.h"

using namespace std;

KNNClassifier::KNNClassifier(unsigned int n_neighbors, std::string weights) {
    set_neighbors(n_neighbors);
    this->weights = weights;
}

void KNNClassifier::set_neighbors(unsigned int k) {
    k_neighbors = k;
}

void KNNClassifier::fit(Matrix X, Matrix y) {
    X_train = X;
    y_train = y;
}

const std::string WEIGHTS_UNIFORM = "uniform";
const std::string WEIGHTS_DISTANCE = "distance";
const std::string WEIGHTS_DISTANCE_POW = "distance_pow";
const std::string WEIGHTS_DIST_POW_5 = "distance_pow_5";
const std::string WEIGHTS_DIST_POW_10 = "distance_pow_10";

Vector KNNClassifier::predict(Matrix X) {
    // Creamos vector columna a devolver
    auto classifications = Vector(X.rows());

    for (unsigned k = 0; k < X.rows(); ++k) {
        // X.row(k) es una imagen que queremos clasificar
        Vector img = X.row(k);

        // Nos quedamos con los K vecinos mas cercanos para cada imagen
        // k vecinos mas cercanos ordenados por distancia de forma ascendente
        OrderedArray arr(this->k_neighbors);

        // Vemos la distancia a cada una de las imagenes del training set
        for(unsigned i = 0; i < X_train.rows(); ++i) {
            Vector img_train = X_train.row(i);

            // Nos basta con tomar la norma al cuadrado, pues
            // la relacion de orden se mantiene.
            double dist = (img - img_train).transpose() * (img - img_train);

            // Insertamos de forma ordenada en k_nearest
            arr.insert(neighbor{dist, int(y_train(i, 0))});
        }

        // Buscamos la moda
        double votes[10] = {0};
        for (neighbor n : arr.k_nearest) {
            // Por si acaso vemos que el digito sea valido
            assert(0 <= n.digit && n.digit < 10);

            // Para distancia, queremos que a menor distancia mas weight.
            // Una forma de hacer esto es que el weight sea 1/dist, ya que
            // no es necesario que el mismo esté normalizado.
            double weight = 0;
            if(this->weights == WEIGHTS_UNIFORM)      weight = 1;
            if(this->weights == WEIGHTS_DISTANCE)     weight = 1/n.dist;
            if(this->weights == WEIGHTS_DISTANCE_POW) weight = 1/pow(n.dist, 3);
            if(this->weights == WEIGHTS_DIST_POW_5)   weight = 1/pow(n.dist, 5);
            if(this->weights == WEIGHTS_DIST_POW_10)  weight = 1/pow(n.dist, 10);

            votes[int(n.digit)] += 1 * weight;
        }

        double max = -1;
        int classification = -1;
        for(int i = 0; i < 10; i++) {
            if (votes[i] > max) { 
                max = votes[i];
                classification = i;
            }
        }

        classifications(k) = classification;
    }

    return classifications;
}

#include <algorithm>
//#include <chrono>
#include <iostream>
#include "knn.h"

using namespace std;


KNNClassifier::KNNClassifier(unsigned int n_neighbors): k_neighbors(n_neighbors) {
}

void KNNClassifier::fit(Matrix X, Matrix y) {
    X_train = X;
    y_train = y;
}

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
            Vector img_train = X_train.row(k);

            // Nos basta con tomar la norma al cuadrado, pues
            // la relacion de orden se mantiene.
            double dist = (img - img_train).transpose() * (img - img_train);

            // Insertamos de forma ordenada en k_nearest
            arr.insert(neighbor{dist, y_train(k, 0)});
        }

        // En arr.k_neighbors tenemos los k vecinos mas cercanos.
        // Buscamos la moda
        int votes[10];
        for (neighbor n : arr.k_nearest) {
            // xlas
            assert(0 <= n.digit && n.digit < 10);
            votes[n.digit]++;
        }

        int max = -1;
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

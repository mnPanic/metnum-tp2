#pragma once

#include "types.h"
#include <vector>
#include <iostream>

class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);

private:

    // matriz de entrenamiento con imagenes
    Matrix X_train;

    // clasificaciones
    Matrix y_train;

    // Cantidad de vecinos a considerar
    int k_neighbors;
};

struct neighbor {
    double dist;
    int digit;
};

// para tests
std::ostream& operator<<(std::ostream& os, const neighbor n) { 
    os << "[" <<  n.digit << "] " << n.dist;
    return os;
}

std::ostream& operator<<(std::ostream& os, const std::vector<neighbor> ns) {
    for(neighbor n : ns) {
        os << ns << "; ";
    }

    return os;
}

struct OrderedArray {
    // ordenado en forma ascendente
    std::vector<neighbor> k_nearest;
    OrderedArray(int k): k_nearest(k, neighbor{INFINITY, 0}) {}

    // Inserta manteniendo el orden, si hace overflow del contenedor
    // descarta el elemento.
    void insert(neighbor n) {
        int k = k_nearest.size();

        if(n.dist >= k_nearest[k - 1].dist) {
            // No nos interesa
            return;
        }

        // Si es menor al ultimo (el mas grande), lo swapeamos y luego
        // lo llevamos al lugar donde pertenece swapeando.
        k_nearest[k - 1] = n;

        for (
            int i = k - 1; 
            i > 0 && k_nearest[i].dist < k_nearest[i - 1].dist;
            i--
        ) {
            swap(k_nearest[i], k_nearest[i - 1]);
        }
    }
};
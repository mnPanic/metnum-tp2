#pragma once

#include "types.h"
#include <vector>
#include <iostream>

class KNNClassifier {
public:
    KNNClassifier(unsigned int n_neighbors);

    void fit(Matrix X, Matrix y);

    Vector predict(Matrix X);

    void set_neighbors(unsigned int k);

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

    bool operator== (const neighbor& other) const {
        return other.digit == digit && other.dist == dist;
    }
};

struct OrderedArray {
    // Ordenado en forma ascendente
    std::vector<neighbor> k_nearest;
    OrderedArray(int k): k_nearest(k, neighbor{INFINITY, 0}) {}

    // swapea en la lista los dos indices indicados
    void swp(int i, int j) {
        neighbor tmp = k_nearest[i];
        k_nearest[i] = k_nearest[j];
        k_nearest[j] = tmp;
    }

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
            swp(i, i - 1);        
        }
    }
};
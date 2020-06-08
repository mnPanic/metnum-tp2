#pragma once
#include "types.h"
#include "eigen.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X, unsigned int n_iter=5000, double eps=1e-5);
private:
    unsigned int alpha;
    Matrix data;
};

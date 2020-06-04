#pragma once
#include "types.h"
#include "eigen.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X);

    Eigen::MatrixXd transform(Matrix X, unsigned num_iter=5000, double epsilon=1e-16);
private:
    unsigned int alpha;
    Matrix data;
};

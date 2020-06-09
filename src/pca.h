#pragma once
#include "types.h"
#include "eigen.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X, unsigned int n_iter=5000, double eps=1e-2);

    Eigen::MatrixXd transform(Matrix X);
private:
    unsigned int alpha;
    Matrix tc;
};

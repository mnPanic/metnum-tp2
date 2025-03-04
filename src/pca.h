#pragma once
#include "types.h"
#include "eigen.h"

class PCA {
public:
    PCA(unsigned int n_components);

    void fit(Matrix X, unsigned int n_iter=5000, double eps=1e-6);

    Eigen::MatrixXd transform(Matrix X);

    void set_tc(Matrix);
    Matrix get_tc();

private:
    unsigned int alpha;
    Matrix tc;
};

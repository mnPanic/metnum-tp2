//
// Created by pachi on 5/6/19.
//

#include <iostream>
#include "pca.h"
#include "eigen.h"

int main(int argc, char** argv){

    Matrix m;
    PCA pca(1);
    std::cout << "===== Random: ======" << std::endl;
    m = Eigen::DiagonalMatrix<double, 3>(3, 8, 6);
    std::cout << "Resultado:\n " << pca.transform(m) << std::endl;

    return 0;
}

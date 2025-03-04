#include <iostream>
#include "pca.h"
#include "eigen.h"

using namespace std;


PCA::PCA(unsigned int n_components) : alpha(n_components) {};

void PCA::fit(Matrix X, unsigned int n_iter, double eps) {

    /// Vector de promedio de las filas de X. mu \in R^{1 x m}///
    Vector mu = (Vector::Constant(X.rows(), 1).transpose() * X)/ X.rows();
    /// Matriz con filas iguales al vector mu. mu_b_r \in R^{}///
    Matrix mu_broadcast_rows = (Vector::Constant(X.rows(), 1) * mu.transpose());
    /// Matriz de covarianza ///
    Matrix M = ((X - mu_broadcast_rows).transpose() * (X - mu_broadcast_rows)) / (X.rows() - 1);

    /// Obtenemos los primeros \alpha autovectores para aplicar la transformación característica.
    std::pair<Vector, Matrix> eigenpair = get_first_eigenvalues(M, this->alpha, n_iter, eps);

    this->tc = eigenpair.second.transpose();
};

void PCA::set_tc(Matrix X) {this->tc = X; };

Matrix PCA::get_tc() { return this->tc; };



MatrixXd PCA::transform(Matrix X) {
    return (this->tc * X.transpose()).transpose();
};

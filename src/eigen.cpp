#include <algorithm>
#include <chrono>
#include <iostream>
#include "eigen.h"

using namespace std;


pair<double, Vector> power_iteration(const Matrix& X, unsigned num_iter, double eps)
{
	Vector b = Vector::Random(X.cols());
	//Calculate iteratively b_k+1 = X*b_k / ||X*b_k||
	while (num_iter-- > 0) {
		Eigen::VectorXd b_k = X * b;

		double b_k_norm = b_k.norm();

		b = b_k / b_k_norm;
	}

	//eigenvalue associated to this eigenvector = (vt A v) / vt v
	double eigval = b.transpose() * X * b;
	eigval = eigval / (b.transpose() * b);
	return make_pair(eigval, b / b.norm());
}

pair<Vector, Matrix> get_first_eigenvalues(const Matrix& X, unsigned num, unsigned num_iter, double epsilon)
{
	Matrix A(X);
	Vector eigvalues(num);
	Matrix eigvectors(A.rows(), num);

	for (unsigned int i = 0; i < num; i++) {
		//Find the eigenvector and the associated eigenvalue
		auto eigen_pair = power_iteration(A, num_iter, epsilon);
		//Assign the ith eigen value to the ith coordinate
		eigvalues(i) = eigen_pair.first;
		//Assign the ith eigen vector to the ith column 
		eigvectors.block(0, i, eigvectors.rows(), 1) = eigen_pair.second;

		//Subatract to matrix A, A - lambda * v * vt
		A -= eigen_pair.first * (eigen_pair.second * eigen_pair.second.transpose());
	}

	return make_pair(eigvalues, eigvectors);
}

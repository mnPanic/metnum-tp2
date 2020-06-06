"""Corre los tests"""

import metnum
import unittest

import numpy as np
import scipy

class TestKNNClassifier(unittest.TestCase):
    def test_same(self):
        """Testea que predecir la matriz de entrenamiento de el mismo res"""
        classifier = metnum.KNNClassifier(1)

        X_train = np.array([
            [1, 2, 3],
            [6, 6, 6],
            [0, 1, 0],
        ])

        y_train = np.array([
            [0],
            [2],
            [3],
        ])

        classifier.fit(X_train, y_train)
        
        y = classifier.predict(X_train)

        self.assertTrue(np.allclose(y.reshape(3, 1), y_train))
    
    def test_little(self):
        """Testea un caso en R2"""
        classifier = metnum.KNNClassifier(3)

        X_train = np.array([
            [0, 0],   [1, 0],   [0, 1],   [1, 1],
            [5, 5],   [6, 5],   [5, 6],   [6, 6],
            [20, 20], [21, 20], [20, 22], [22, 20],
        ])

        y_train = np.array([
            [0], [0], [0], [0],
            [5], [5], [5], [5],
            [9], [9], [9], [9],
        ])

        classifier.fit(X_train, y_train)

        X = np.array([
            [15, 15], # 9
            [10, 10], # 5
            [2, 2],   # 0
        ])

        want = np.array([9.0, 5.0, 0.0])
        got = classifier.predict(X)

        assertAllClose(self, want, got)

class TestPowerIteration(unittest.TestCase):
    def assertEigenpair(self, X, a, v):
        assertAllClose(self, np.abs(X @ v), np.abs(a * v))

    def test_diagonal(self):
        X = np.diag([3, -10, 2])
        a, v = metnum.power_iteration(X)
        v = v.reshape(3, 1)
        self.assertEigenpair(X, a, v)

    def test_single_item(self):
        X = np.array([20])
        a, v = metnum.power_iteration(X)
        self.assertEigenpair(X, a, v)

def assertAllClose(self, want, got):
    """Se fija que dos vectores esten np.allclose"""
    self.assertTrue(np.allclose(want, got), f"want: {want}, but got: {got}")

if __name__ == '__main__':
    unittest.main()
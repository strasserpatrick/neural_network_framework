import numpy as np


class PCA:
    def __init__(self, num_components):
        self.num_components = num_components
        self.eigenvector_subset = None

    def fit_transform(self, X):
        self.mean = np.mean(X, axis=0)
        X_meaned = X - self.mean

        cov_mat = np.cov(X_meaned, rowvar=False)

        eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        self.eigenvector_subset = sorted_eigenvectors[:, 0 : self.num_components]

        used_variance = np.sum(sorted_eigenvalue[0 : self.num_components])
        total_variance = np.sum(sorted_eigenvalue)
        explained_variance = used_variance / total_variance

        X_reduced = np.dot(
            self.eigenvector_subset.transpose(), X_meaned.transpose()
        ).transpose()

        return X_reduced, explained_variance

    def transform(self, X):
        X_meaned = X - self.mean
        eigenvector_subset = self.eigenvector_subset
        X_reduced = np.dot(
            eigenvector_subset.transpose(), X_meaned.transpose()
        ).transpose()
        return X_reduced

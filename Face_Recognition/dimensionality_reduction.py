import numpy as np

class PCA():
    def __init__(self, n_components):
        self.n_components = n_components
    
    def fit(self, X):
        """
        Method which calculates eigen vectors and their projections in the eigen space of given data covariance matrix
        X: input data of faces
        returns:
        - C: principal components which are top n eigen vectors
        - P: projection of these eigenvectors into eigen space
        - M: mean vector of input data
        - N: normalized input data
        """
        self.mean_vector = X.mean(axis=1)
        self.mean_vector = self.mean_vector.reshape(X.shape[0], 1)
        normalized_data = X - self.mean_vector
        cov = np.cov(normalized_data.T)
        eig_val, eig_vec = np.linalg.eig(cov)
        idx = eig_val.argsort()[::-1]   
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:,idx]
        # print(eig_val[:10])
        # print(eig_val[:self.n_components])
        # print(eig_val[-self.n_components:-1])
        self.eig_vectors = (eig_vec[:self.n_components]).real
        self.eig_faces = self.eig_vectors.dot(normalized_data.T).real
        # print(self.eig_vectors.shape, self.eig_faces.shape)
        return (self.eig_vectors, self.eig_faces)

    def transform(self, X):
        """
        Method which reduces the given data to k dimensional format
        X: input data of faces
        returns:
        - reduces_data: which is lower dimensional representation of given data in higher dimension
        """
        normalized_data = X - self.mean_vector
        reduced_data = normalized_data.T.dot(self.eig_faces.T)
        return reduced_data

    def inverse_transform(self, X):
        """
        Method which inverse transforms lower dimensional data to higher dimensional data
        X: input data of faces
        returns:
        - original_data: which is higher dimensional representation of given data (original of reduced)
        """
        # print(X.shape, self.eig_faces.shape, self.mean_vector.shape)
        return X.dot(self.eig_faces) + self.mean_vector.T
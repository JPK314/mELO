import numpy as np
import torch

class mELO:
    def __init__(self, n: int, k: int, lr_ratings: float=16, lr_features: float=1, initial_ratings: np.ndarray=None, initial_features: np.ndarray=None):
        """
        :param n: number of players in the elo pool.
        :param lr_ratings: learning rate for ratings. Default 16.
        :param lr_features: learning rate for features. Default 1.
        :param k: (half of) the dimension of the feature vector. k=0 is equivalent to standard ELO.
        :param initial_ratings: a vector of length n such that the ith value is the rating of the ith player.
        :param inital_features: a (n,2k)-shaped array of features such that the ith index is the feature vector for the ith player.
        """
        if initial_ratings == None:
            initial_ratings = np.zeros(n)
        if initial_features == None:
            initial_features = np.zeros((n,2*k))
        self.k = k
        self.lr_ratings = lr_ratings
        self.lr_features = lr_features
        self.ratings = initial_ratings
        self.features = initial_features
        self.omega = np.zeros((2*k,2*k))
        for i in range(k):
            self.omega[2*i][2*i+1] = 1
            self.omega[2*i+1][2*i] = -1
    
    def get_ratings(self):
        return self.ratings.copy()
    
    def get_features(self):
        return self.features.copy()

    def perform_update(self, i: int, j: int, outcome: float):
        """
        :param i: index of first player
        :param j: index of second player
        :param outcome: observed probability that i beats j
        """
        # Due to how atleast_2d works, this is already transposed (a row vector)
        c_i_transpose = np.atleast_2d(self.features[i])
        # and similarly, this one needs to be transposed (to a column vector)
        c_j = np.atleast_2d(self.features[j]).T
        p_hat_ij = torch.sigmoid(self.ratings[i]-self.ratings[j]+np.matmul(np.matmul(c_i_transpose, self.omega), c_j)[0][0])
        delta = outcome - p_hat_ij
        self.ratings[i] += self.lr_ratings * delta
        self.ratings[j] -= self.lr_ratings * delta
        self.features[i] += self.lr_features * delta * np.matmul(self.omega, self.features[i])
        self.features[j] -= self.lr_features * delta * np.matmul(self.omega, self.features[j])


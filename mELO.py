import numpy as np
import sqlite3 as sl

def sigmoid(x):
 return 1/(1 + np.exp(-x))

class mELO:
    def __init__(self, n: int, k: int, lr_ratings: float=16, lr_features: float=1, initial_ratings: np.ndarray=None, initial_features: np.ndarray=None):
        """
        :param n: number of players in the elo pool.
        :param k: (half of) the dimension of the feature vector. k=0 is equivalent to standard ELO.
        :param lr_ratings: learning rate for ratings. Default 16.
        :param lr_features: learning rate for features. Default 1.
        :param initial_ratings: a vector of length n such that the ith value is the rating of the ith player.
        :param inital_features: a (n,2k)-shaped array of features such that the ith index is the feature vector for the ith player.
        """
        if initial_ratings == None:
            initial_ratings = np.zeros(n)
        if initial_features == None:
            initial_features = np.random.normal(loc = 0, scale = 1, size = (n, 2 * k))
            # orthogonalize
            initial_features, R = np.linalg.qr(initial_features)

        self.k = k
        self.n = n
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
        p_hat_ij = sigmoid(self.ratings[i]-self.ratings[j]+np.matmul(np.matmul(c_i_transpose, self.omega), c_j)[0][0])
        delta = outcome - p_hat_ij
        derivative_coeff = delta * p_hat_ij * (1 - p_hat_ij)
        self.ratings[i] += self.lr_ratings * derivative_coeff
        self.ratings[j] += self.lr_ratings * -derivative_coeff
        old_features = self.features.copy()
        for q in range(self.k):
            self.features[i][2 * q] += self.lr_features * derivative_coeff * old_features[j][2 * q + 1]
            self.features[i][2 * q + 1] += self.lr_features * derivative_coeff * -old_features[j][2 * q]
            self.features[j][2 * q] += self.lr_features * derivative_coeff * -old_features[i][2 * q + 1]
            self.features[j][2 * q + 1] += self.lr_features * derivative_coeff * old_features[i][2 * q]

    def get_P(self):
        """
        Returns nxn matrix P where P[i,j] is the probability that i beats j
        """
        p = np.zeros((self.n,self.n))
        for i in range(self.n):
            c_i_transpose = np.atleast_2d(self.features[i])
            p[i] = sigmoid(self.ratings[i] * np.ones(self.n) - self.ratings + np.matmul(np.matmul(c_i_transpose, self.omega), self.features.T))
        return p

    def save(self, ratings_db: str, ratings_table: str, agents: str):
        """
        Save ratings to database.
        :param ratings_db: database file name
        :param ratings_table: database table name
        :param agents: list of agents such that agent[i] has ratings[i] and features[i]
        """
        con = sl.connect(ratings_db)
        cur = con.cursor()
        table_name = cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ratings_table}'").fetchone()
        if table_name != None:
            cur.execute(f"DROP TABLE {ratings_table}")
        cur.execute(f"CREATE TABLE {ratings_table}(agent varchar(255), rating float(24), feature varchar(2047))")
        for idx in range(self.n):
            cur.execute(f"INSERT INTO {ratings_table} VALUES ('{agents[idx]}', {str(self.ratings[idx])}, '{','.join([str(v) for v in self.features[idx]])}')")
        con.commit()
    
    def load(self, ratings_db: str, ratings_table: str):
        """
        Load ratings from database, ordered by agent name (ascending).
        :param ratings_db: database file name
        :param ratings_table: database table name
        """
        con = sl.connect(ratings_db)
        cur = con.cursor()
        table_name = cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ratings_table}'").fetchone()
        if table_name == None:
            raise Exception("No table found to load from")
        agents_ratings_features = self.cur.execute(f"SELECT * FROM {ratings_table} ORDER BY agent ASC").fetchall()
        n = len(agents_ratings_features)
        if n == 0:
            raise Exception("Table has no data")
        if n != self.n:
            print("Warning: the number of agents in the database is different from what was used to initialize. Overwriting.")
            self.n = n
            self.ratings = np.zeros(self.n)
        k = len(np.fromstring(agents_ratings_features[0][2]))
        if k != self.k:
            print("Warning: the value for k in the features in the database is different from what was used to initialize. Overwriting.")
            self.k = k
            initial_features = np.random.normal(loc = 0, scale = 1, size = (self.n, 2 * self.k))
            self.features, R = np.linalg.qr(initial_features)

        for idx, (a, r, f) in enumerate(agents_ratings_features):
            self.ratings[idx] = float(r)
            self.features[idx] = np.fromstring(f, sep=",")
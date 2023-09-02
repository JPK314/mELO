import numpy as np
import sqlite3 as sl

def sigmoid(x):
 return 1/(1 + np.exp(-x))

class mELO_db:
    def __init__(self, k: int, ratings_db: str, ratings_table: str, lr_ratings: float=16, lr_features: float=1, initialize=False, agents=[]):
        """
        :param k: (half of) the dimension of the feature vector. k=0 is equivalent to standard ELO.
        :param ratings_db: a name for the sqlite3 database in which the ratings are stored.
        :param ratings_table: the table used inside the ratings db.
        :param lr_ratings: learning rate for ratings. Default 16.
        :param lr_features: learning rate for features. Default 1.
        :param inital_features: a (n,2k)-shaped array of features such that the ith index is the feature vector for the ith player.
        """
        self.con = sl.connect(ratings_db)
        self.cur = self.con.cursor()
        table_name = self.cur.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{ratings_table}'").fetchone()
        if initialize or table_name == None:
            self.con = sl.connect(ratings_db)
            self.cur = self.con.cursor()
            if table_name != None:
                self.cur.execute(f"DROP TABLE {ratings_table}")
            self.cur.execute(f"CREATE TABLE {ratings_table}(agent varchar(255), rating float(24), feature varchar(2047))")
            features = np.random.normal(loc = 0, scale = 5, size = (len(agents), 2 * k))
            for idx, agent in enumerate(agents):
                self.cur.execute(f"INSERT INTO {ratings_table} VALUES ('{agent}', 0, '{','.join([str(v) for v in features[idx]])}')")
            self.con.commit()

        self.k = k
        self.ratings_table = ratings_table
        self.lr_ratings = lr_ratings
        self.lr_features = lr_features
        self.omega = np.zeros((2*k,2*k))
        for i in range(k):
            self.omega[2*i][2*i+1] = 1
            self.omega[2*i+1][2*i] = -1

    def perform_update(self, a1: str, a2: str, outcome: float):
        """
        :param a1: name of first player in database
        :param a2: name of second player in database
        :param outcome: observed probability that i beats j
        """
        (r1str, f1str) = self.cur.execute(f"SELECT rating, feature FROM {self.ratings_table} WHERE agent='{a1}'").fetchone()
        (r2str, f2str) = self.cur.execute(f"SELECT rating, feature FROM {self.ratings_table} WHERE agent='{a2}'").fetchone()
        r1 = float(r1str)
        r2 = float(r2str)
        f1 = np.fromstring(f1str, sep=",")
        f2 = np.fromstring(f2str, sep=",")
        # Due to how atleast_2d works, this is already transposed (a row vector)
        c_i_transpose = np.atleast_2d(f1)
        # and similarly, this one needs to be transposed (to a column vector)
        c_j = np.atleast_2d(f2).T
        p_hat_ij = sigmoid(r1-r2+np.matmul(np.matmul(c_i_transpose, self.omega), c_j)[0][0])
        delta = outcome - p_hat_ij
        derivative_coeff = delta * p_hat_ij * (1 - p_hat_ij)
        r1 += self.lr_ratings * derivative_coeff
        r2 += self.lr_ratings * -derivative_coeff
        old_f1 = f1.copy()
        old_f2 = f2.copy()
        for q in range(self.k):
            f1[2 * q] += self.lr_features * derivative_coeff * old_f2[2 * q + 1]
            f1[2 * q + 1] += self.lr_features * derivative_coeff * -old_f2[2 * q]
            f2[2 * q] += self.lr_features * derivative_coeff * -old_f1[2 * q + 1]
            f2[2 * q + 1] += self.lr_features * derivative_coeff * old_f1[2 * q]
        self.cur.execute(f"UPDATE {self.ratings_table} SET rating={r1}, feature='{','.join([str(v) for v in f1])}' WHERE agent='{a1}'")
        self.cur.execute(f"UPDATE {self.ratings_table} SET rating={r2}, feature='{','.join([str(v) for v in f2])}' WHERE agent='{a2}'")
        self.con.commit()

    def get_P(self):
        """
        Returns nxn matrix P where P[i,j] is the probability that i beats j
        """
        agents_ratings_features = self.cur.execute(f"SELECT * FROM {self.ratings_table} ORDER BY agent ASC").fetchall()
        n = len(agents_ratings_features)
        ratings = np.array([float(r) for (a,r,f) in agents_ratings_features])
        features = np.zeros((n, 2 * self.k))
        for idx, (a,r,f) in enumerate(agents_ratings_features):
            features[idx] = np.fromstring(f, sep=",")
        p = np.zeros((n,n))
        for i in range(n):
            c_i_transpose = np.atleast_2d(features[i])
            p[i] = sigmoid(ratings[i] * np.ones(n) - ratings + np.matmul(np.matmul(c_i_transpose, self.omega), features.T))
        return p


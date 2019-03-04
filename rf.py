import pandas as pd
import numpy as np


class TreeEnsemble():
    def __init__(self, x, y, n_trees, sample_size, min_leaf=5):
        np.random.seed(15)
        self.x = x
        self.y = y
        self.sample_size = sample_size
        self.min_leaf = min_leaf
    
    def create_tree(self):
        rnd_idxs = np.random.permutation(len(self.y))[:self.sample_size]
        return DecisionTree(self.x.iloc[rnd_idxs], self.y[rnd_idxs], min_leaf=self.min_leaf)

    def predict(self, x):
        return np.mean([t.predict for t in self.trees], axis=0)

    

class DecisionTree():
    pass
import numpy as np
import pandas as np
import math, random
from sklearn import metrics
from concurrent.futures import ProcessPoolExecutor


class TreeEnsemble():
    def __init__(self, x, y, n_trees, sample_sz, min_leaf=5, max_features=None):
        np.random.seed(42)
        self.x = x.values
        self.x_unuse1 = x
        self.y = y
        self.sample_sz = sample_sz
        self.min_leaf = min_leaf
		self.max_features = 1 if max_features is None else max_features
        self.trees = [self.create_tree() for i in range(n_trees)]

    def create_tree(self):
        idxs = np.random.permutation(len(self.y))[:self.sample_sz]
        
        return DecisionTree(self.x[idxs], self.y[idxs], 
                    idxs=np.array(range(self.sample_sz)),
                    x_unuse=self.x_unuse1,
                    min_leaf=self.min_leaf, 
                    max_features=self.max_features)
        
    def predict(self, x):
        return np.mean([t.predict(x) for t in self.trees], axis=0)
    
	
    def feature_importance(self):
        x_imp = self.x.copy()        
        tmp = self.predict(x_imp)        
        start_score = metrics.r2_score(self.y, tmp)
        importance = dict()
        
        for i, n in zip(range(self.c), list(X_train.columns)):
            now = x_imp[:,i]
            x_imp[:,i] = np.random.permutation(now)
            end_score = metrics.r2_score(self.y, self.predict(x_imp))
            importance[n] = start_score - end_score           
            x_imp[:,i] = now
        return sorted(importance.items(), key=lambda kv: kv[1], reverse=True)

    def confidence_on_variance(self, x):
        def get_predicts(t): return t.predict(x)
        pred = np.stack(map(get_predicts, self.trees))
        self.pred_mean = np.mean(pred[:,0])
        self.pred_std = np.std(pred[:,0])        
        print(f"Mean is {self.pred_mean}.\nStd is {self.pred_std}")
    

def std_agg(num_sampl, s1, s2): return math.sqrt((s2/num_sampl) - (s1/num_sampl)**2)

class DecisionTree():
    def __init__(self, x, y, idxs, max_features, x_unuse=None, min_leaf=5):
        self.x = x
        self.y = y
        self.idxs = idxs
        self.min_leaf = min_leaf
        self.n = len(idxs)
        self.c = x.shape[1]
        self.val = np.mean(y[idxs])
        self.max_features = max_features
        self.x_unuse = x_unuse
        self.score = float('inf')
        self.find_varsplit()
        
    def find_varsplit(self):
        # max features
        lenght = [i for i in range(self.c)]
        random.shuffle(lenght)
        rg = int(len(lenght) * self.max_features)

        for i in lenght[:rg]: self.find_better_split(i)        
        if self.score == float('inf'): return
        x = self.split_col
        lhs = np.nonzero(x <= self.split)[0]
        rhs = np.nonzero(x >  self.split)[0]
        self.lhs = DecisionTree(self.x, self.y, self.idxs[lhs], self.max_features)
        self.rhs = DecisionTree(self.x, self.y, self.idxs[rhs], self.max_features)

    def find_better_split(self, var_idx):
        x = self.x[self.idxs,var_idx]
        y =  self.y[self.idxs]
        sort_idx = np.argsort(x)
        sort_y = y[sort_idx]
        sort_x = x[sort_idx]
        rhs_sampl, rhs_sum, rhs_sum2 = self.n, sort_y.sum(), (sort_y**2).sum()
        lhs_sampl, lhs_sum, lhs_sum2 = 0, 0., 0.

        for i in range(0, self.n-self.min_leaf):
            xi = sort_x[i]
            yi = sort_y[i]
            lhs_sampl += 1;     rhs_sampl -= 1
            lhs_sum   += yi;    rhs_sum -= yi
            lhs_sum2  += yi**2; rhs_sum2 -= yi**2
            if i < self.min_leaf-1 or xi == sort_x[i+1]:
                continue

            lhs_std = std_agg(lhs_sampl, lhs_sum, lhs_sum2)
            rhs_std = std_agg(rhs_sampl, rhs_sum, rhs_sum2)
            curr_score = lhs_std * lhs_sampl + rhs_std * rhs_sampl
            if curr_score < self.score: 
                self.var_idx, self.score, self.split = var_idx, curr_score, xi

    @property
    def split_name(self): return self.x_unuse.columns[self.var_idx]
    
    @property
    def split_col(self): return self.x[self.idxs,self.var_idx]

    @property
    def is_leaf(self): return self.score == float('inf')
    
    def __repr__(self):
        s = f'n: {self.n}; val:{self.val}'
        if not self.is_leaf:
            s += f'; score:{self.score}; split:{self.split}; var:{self.split_name}'
        return s

    def predict(self, x):
        return np.array([self.predict_row(xi) for xi in x])

    def predict_row(self, xi):
        if self.is_leaf: return self.val
        t = self.lhs if xi[self.var_idx] <= self.split else self.rhs
        return t.predict_row(xi)

import sys
import numpy as np
import pandas as pd
import multiprocessing
from copy import deepcopy as copy
from joblib import Parallel, delayed
from tqdm.notebook import tqdm as tqdm_n
from sklearn.metrics import roc_auc_score, log_loss

class FGBClassifier():
    
    ##############
    # TODO:
    # bootstrap
    # max_depth
    # max_features
    ##############
    
    def __init__(self, n_estimators=10, learning_rate=1e-2, theta=0.5, verbose=True):
        self.theta = theta
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate

    def fit(self, X, y, s):

        def get_batches(iterable, n_jobs=-1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count() # - 1 # -1 so that our laptop doesn't freeze
            l = len(iterable)
            n = int(np.ceil(l / n_jobs))
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        def find_best_split_parallel(batch, X, y, s, p, idx, theta, learning_rate):
            base_score = 0.5 - theta
            best_score = copy(base_score)
            ovr_weights = np.repeat(1.0, s.shape[1])
            for split in batch:
                variable, value = split
                left_idx = idx[X[idx, variable]<value]
                right_idx = idx[X[idx, variable]>=value]
                left_n, right_n = len(left_idx), len(right_idx)
                if (left_n>0) and (right_n>0):
                    left_y, right_y = y[left_idx], y[right_idx]
                    left_s, right_s = s[left_idx], s[right_idx]
                    left_p, right_p = p[left_idx], p[right_idx]
                    for j in range(s.shape[1]):
                        left_swap_s = np.argmax([
                            log_loss(left_s[:,j], left_p, labels=[0,1]), 
                            log_loss(1-left_s[:,j], left_p, labels=[0,1]), 
                        ])
                        if left_swap_s:
                            left_s[:,j] = 1-left_s[:,j]
                        right_swap_s = np.argmax([
                            log_loss(right_s[:,j], right_p, labels=[0,1]), 
                            log_loss(1-right_s[:,j], right_p, labels=[0,1]), 
                        ])
                        if right_swap_s:
                            right_s[:,j] = 1-right_s[:,j]
                    left_p_increase = np.mean(
                        -(
                            (
                                (left_s.shape[1]*theta - left_s.shape[1])*left_y + \
                                -1*np.sum(left_s, axis=1) * theta + \
                                left_s.shape[1] * left_p
                            ) / (
                                left_s.shape[1]
                            )
                        )
                    )*learning_rate
                    right_p_increase = np.mean(
                        -(
                            (
                                (right_s.shape[1]*theta - right_s.shape[1])*right_y + \
                                -1*np.sum(right_s, axis=1) * theta + \
                                right_s.shape[1] * right_p
                            ) / (
                                right_s.shape[1]
                            )
                        )
                    )*learning_rate
                    left_new_p = left_p + left_p_increase
                    right_new_p = right_p + right_p_increase
                    y_auc = roc_auc_score(y[left_idx].tolist() + y[right_idx].tolist(), left_new_p.tolist()+right_new_p.tolist())
                    ovr_s_auc = []
                    for j in range(s.shape[1]):
                        s_auc = roc_auc_score(
                            s[left_idx, j].tolist()+s[right_idx, j].tolist(),
                            left_new_p.tolist()+right_new_p.tolist()
                        )
                        s_auc = max(1-s_auc, s_auc)
                        ovr_s_auc.append(s_auc)
                    s_auc = np.average(ovr_s_auc, weights=ovr_weights)
                    score = (1-theta)*y_auc - theta*s_auc
                    if score > best_score:
                        best_split = split
                        best_score = score
                        best_left_n = left_n
                        best_right_n = right_n
                        best_left_idx = left_idx
                        best_right_idx = right_idx
                        best_left_p_increase = left_p_increase
                        best_right_p_increase = right_p_increase
            if best_score==base_score:
                best_split = np.nan
                best_score = -np.inf
                best_left_idx = np.nan
                best_right_idx = np.nan
                best_left_p_increase = np.nan
                best_right_p_increase = np.nan
            return best_left_p_increase, best_left_idx, best_right_p_increase, best_right_idx, best_split, best_score

        theta = self.theta
        verbose = self.verbose
        n_trees = self.n_estimators
        learning_rate = self.learning_rate

        n, m = X.shape
        p = np.repeat(0.5, n)
        idx = np.array(range(n))
        s = pd.get_dummies(s).values if (len(s.shape)==1) else s

        splits = [
            (variable, np.unique(X[idx, variable])[i])
            for variable in range(m)
                for i in range(len(np.unique(X[idx, variable])))
        ]
        batches = list(get_batches(splits))

        trees = []
        best_score=0
        while best_score!=-np.inf:
            for i in tqdm_n(range(n_trees)):
                results = Parallel(n_jobs=-1)(
                    delayed(find_best_split_parallel)(
                        batch, X, y, s, p, idx, theta, learning_rate
                    ) for batch in batches
                )
                best_left_p_increase, best_left_idx, best_right_p_increase, best_right_idx, best_split, best_score = sorted(
                    results, key=lambda x: x[-1]
                )[-1]
                if best_score!=-np.inf:
                    tree = {
                        "split": best_split,
                        0: best_left_p_increase,
                        1: best_right_p_increase,
                    }
                    trees.append(tree)
                    p[best_left_idx] = p[best_left_idx] + best_left_p_increase
                    p[best_right_idx] = p[best_right_idx] + best_right_p_increase
                    if verbose:
                        y_auc = round(roc_auc_score(
                            y[best_left_idx].tolist()+y[best_right_idx].tolist(),
                            p[best_left_idx].tolist()+p[best_right_idx].tolist(),
                        ), 4)

                        ovr_s_auc = []
                        for j in range(s.shape[1]):
                            s_auc = roc_auc_score(
                                s[best_left_idx, j].tolist()+s[best_right_idx, j].tolist(),
                                p[best_left_idx].tolist()+p[best_right_idx].tolist(),
                            )
                            s_auc = max(1-s_auc, s_auc)
                            ovr_s_auc.append(s_auc)

                        ovr_weights = np.repeat(1.0, s.shape[1])
                        s_auc = round(np.average(ovr_s_auc, weights=ovr_weights), 4)
                        print_line = "y_AUC = " + str(y_auc) + "\ts_AUC = " + str(s_auc)
                        for j in range(s.shape[1]):
                            print_line += "\ts"+str(j+1)+"_AUC = " + str(round(ovr_s_auc[j], 4))
                        sys.stdout.write("\r" + str(print_line)+"\t")
                        sys.stdout.flush()
                else:
                    break
            best_score=-np.inf

        self.trees = trees
    
    def predict_proba(self, X):
        p = np.repeat(0.5, X.shape[0])
        idx = np.array(range(X.shape[0]))
        for tree in self.trees:
            feature, value = tree["split"]
            left_p_increase = tree[0]
            right_p_increase = tree[1]

            left_idx = idx[X[idx, feature]<value]
            right_idx = idx[X[idx, feature]>=value]

            p[left_idx] = p[left_idx] + left_p_increase
            p[right_idx] = p[right_idx] + right_p_increase
        
        proba = p.reshape(-1,1)
        return np.concatenate((1-proba, proba), axis=1)

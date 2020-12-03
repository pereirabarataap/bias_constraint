import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from math import ceil
import multiprocessing
from random import seed
from copy import deepcopy as copy
from scipy.optimize import minimize
from joblib import delayed, Parallel
from scipy.special import expit as sigmoid

class BiasConstraintLogisticRegression():
    
    def __init__(self, ortho=1, l1_reg_factor=0, add_intersect=True, ortho_method="avg", tol=1e-4, maxiter=1e4, random_state=42):
        """
        Logistic Regression with bias constraint
        
        ortho -> float:
            strength of constraint over sensitive feature
            
        ortho_method -> str:["avg", "w_avg", "inv_w_avg", "max"]
            how to compute 1 sens-reg value from categorical sens attribute
            
        random_state -> int:
            random state for initial guess over vector w + b
            
        add_intersect -> bool:
            if True, no intercept is computed (b = 0)
            
        l1_reg_factor -> float:
            proportional to l1-reg strength
            min_gain_increase = l1_reg_factor/X.shape[1]
            defines the minimum relative weight increase to retain coefs      
            
        tol -> float:
            minimum loss change tolerance for termination
            
        maxiter -> int:
            maximum number of iterations to perform
        """
        
        self.tol=tol
        self.ortho=ortho
        self.is_fit=False
        self.maxiter=maxiter
        self.ortho_method=ortho_method
        self.random_state=random_state
        self.add_intersect=add_intersect
        self.l1_reg_factor=l1_reg_factor
        
        seed(random_state)
        np.random.seed(random_state)

    def fit(self, X, y, s):
        def loss(coefs, X, y, s, ortho=1, add_intersect=True, ortho_method="avg"):
            def corr2_coeff(A, B):
                # Rowwise mean of input arrays & subtract from input arrays themeselves
                A_mA = A - A.mean(1)[:, None]
                B_mB = B - B.mean(1)[:, None]

                # Sum of squares across rows
                ssA = (A_mA**2).sum(1)
                ssB = (B_mB**2).sum(1)

                # Finally get corr coeff
                return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))
            
            if add_intersect:
                b = coefs[-1]
                w = coefs[:-1]
            else:
                b = 0
                w = coefs[:]
            score = np.dot(X,w)+b
            pred = sigmoid(score)
            
            #cap in official Kaggle implementation
            #per forums/t/1576/r-code-for-logloss
            epsilon = 1e-15
            pred = np.maximum(epsilon, pred)
            pred = np.minimum(1-epsilon, pred)
            
            loss = sum(y*np.log(pred) + np.subtract(1,y)*np.log(np.subtract(1,pred))) * -1.0 / len(y)
            
            if ortho == 0:
                sens_reg = 0
            else:
                score = score.reshape(len(score), 1)
                if len(np.unique(score))==1:
                    sens_reg = 1
                else:
                    correlations = abs(corr2_coeff(s.T,score.T).ravel())
                    if ortho_method=="max":
                        sens_reg = max(correlations)
                    elif ortho_method=="avg":
                        sens_reg = np.mean(correlations)
                    elif ortho_method=="w_avg":
                        sens_reg = np.average(correlations, weights=np.sum(s, axis=0))
                    elif ortho_method=="inv_w_avg":
                        sens_reg = np.average(correlations, weights=1/np.sum(s, axis=0))
            
            total_loss = loss + ortho*sens_reg
            return total_loss
        
        seed(self.random_state)
        np.random.seed(self.random_state)
        X = np.array(X).astype(float)
        y = np.array(y).astype(int)
        s = np.array(s).astype(str)
        s = pd.get_dummies(s).values.astype(int)
        if self.add_intersect:
            if self.random_state == None:
                coefs = np.ones(shape=(X.shape[1]+1))
            else:
                coefs = np.random.normal(size=(X.shape[1]+1))
        else:
            if self.random_state == None:
                coefs = np.zeros(shape=(X.shape[1]))
            else:
                coefs = np.random.normal(size=(X.shape[1]))
        
        #message = "failure"
        #while "success" not in message:
        result = minimize(
            fun=loss,
            x0=coefs,
            args=(X, y, s, self.ortho, self.add_intersect, self.ortho_method),
            method="SLSQP",
            tol=self.tol,
            options=dict(
                ftol=self.tol,
                maxiter=self.maxiter,
            ),
        )
            #message = result.message
        coefs = result.x
            
        if self.add_intersect:
            self.w = coefs[:-1]
            self.b = coefs[-1]
        else:
            self.w = coefs
            self.b = 0
        
        if self.l1_reg_factor > 0:
            # get useful weights 
            i = 0
            stop = False
            min_gain_increase = self.l1_reg_factor/X.shape[1]
            cum_sum_normalised_w = np.cumsum(np.array(sorted(abs(self.w), reverse=True)) / sum(abs(self.w)))
            while (not stop) and (i < len(self.w)-1):
                gain = cum_sum_normalised_w[i+1] - cum_sum_normalised_w[i] 
                if gain >= min_gain_increase: # if gain in cumdensity of weight is lower than proportion of weight
                    i+=1
                else:
                    stop=True
            retained_w_index = sorted(np.argsort(abs(self.w))[-i-1:])
            X_retained = X[:,retained_w_index]
            if self.add_intersect:
                coefs = np.array((self.w[retained_w_index]).tolist() + [self.b])
            else:
                coefs = self.w[retained_w_index]
            #message = "failure"
            #while "success" not in message:
            result = minimize(
                fun=loss,
                x0=coefs,
                args=(X_retained, y, s, self.ortho, self.add_intersect, self.ortho_method),
                method="SLSQP",
                tol=self.tol,
                options=dict(
                    ftol=self.tol,
                    maxiter=self.maxiter,
                ),
            )
                #message = result.message
            coefs = result.x
            
            if self.add_intersect:
                self.b = coefs[-1]
                w = np.zeros(shape=X.shape[1])
                w[retained_w_index] = coefs[:-1]
                self.w = w
                
            else:
                self.b = 0
                w = np.zeros(shape=X.shape[1])
                w[retained_w_index] = coefs[:]
                self.w = w
            
            self.is_fit=True
        else:
            self.is_fit=True
    
    def predict(self, X):
        """
        Returns raw logit score (-int, +inf)
        """
        return np.dot(X, self.w) + self.b
    
    def predict_proba(self, X):
        return np.array(
            [
                1-sigmoid(np.dot(X, self.w) + self.b),
                sigmoid(np.dot(X, self.w) + self.b),
            ]
        ).T

class BiasConstraintLogisticRegression2():
    
    def __init__(self, ortho=0, l1_reg_factor=0, ortho_method="max", add_intersect=True, tol=1e-6, maxiter=1e6):
        """
        Logistic Regression with bias constraint
        
        ortho -> float:
            strength of constraint over sensitive feature
            
        ortho_method -> str:["avg", "w_avg", "inv_w_avg", "max"]
            how to compute 1 sens-reg value from categorical sens attribute
            
            
        add_intersect -> bool:
            if True, no intercept is computed (b = 0)
            
        l1_reg_factor -> float:
            proportional to strength of squishing coefs      
            
        tol -> float:
            minimum loss change tolerance for termination
            
        maxiter -> int:
            maximum number of iterations to perform
        """
        
        self.tol=tol
        self.ortho=ortho
        self.is_fit=False
        self.maxiter=maxiter
        self.ortho_method=ortho_method
        self.l1_reg_factor=l1_reg_factor
        self.add_intersect=add_intersect
        
    def fit(self, X, y, s):
        def loss(coefs, X, y, s, ortho=0, l1_reg_factor=0, ortho_method="avg", add_intersect=True):
            if add_intersect:
                b = coefs[-1]
                w = coefs[:-1]
            else:
                b = 0
                w = coefs[:]
            score = np.dot(X,w)+b
            pred = sigmoid(score)
            
            #cap in official Kaggle implementation
            #per forums/t/1576/r-code-for-logloss
            epsilon = 1e-15
            pred = np.maximum(epsilon, pred)
            pred = np.minimum(1-epsilon, pred)
            
            loss = sum(y*np.log(pred) + np.subtract(1,y)*np.log(np.subtract(1,pred))) * -1.0 / len(y)
            
            l1_reg = 0
            if l1_reg_factor > 0:
                l1_reg = np.linalg.norm(w, ord=1)
            
            sens_reg = 0
            if ortho > 0:
                s_means = np.mean(s, axis=0)
                scores = np.repeat(score, s.shape[1]).reshape(s.shape)
                sens_vec = np.mean(abs((s - s_means) * scores), axis=0)
                if ortho_method=="max":
                    sens_reg = max(sens_vec)
                elif ortho_method=="avg":
                    sens_reg = np.mean(sens_vec)
                elif ortho_method=="w_avg":
                    sens_reg = np.average(sens_vec, weights=np.sum(s, axis=0))
                elif ortho_method=="inv_w_avg":
                    sens_reg = np.average(sens_vec, weights=1/np.sum(s, axis=0))
            
            total_loss = loss + l1_reg*l1_reg_factor + ortho*sens_reg
            return total_loss
        
        X = np.array(X).astype(float)
        y = np.array(y).astype(int)
        s = np.array(s).astype(str)
        s = pd.get_dummies(s).values.astype(int)
        if self.add_intersect:
            coefs = np.zeros(shape=(X.shape[1]+1))
        else:
            coefs = np.zeros(shape=(X.shape[1]))

        result = minimize(
            fun=loss,
            x0=coefs,
            args=(X, y, s, self.ortho, self.l1_reg_factor, self.ortho_method, self.add_intersect),
            method="SLSQP",
            tol=self.tol,
            options=dict(
                ftol=self.tol,
                maxiter=self.maxiter,
            ),
        )
        coefs = result.x
            
        if self.add_intersect:
            self.w = coefs[:-1]
            self.b = coefs[-1]
        else:
            self.w = coefs
            self.b = 0

        self.is_fit=True
    
    def predict(self, X):
        """
        Returns raw logit score (-int, +inf)
        """
        return np.dot(X, self.w) + self.b
    
    def predict_proba(self, X):
        return np.array(
            [
                1-sigmoid(np.dot(X, self.w) + self.b),
                sigmoid(np.dot(X, self.w) + self.b),
            ]
        ).T
    
class BiasConstraintDecisionTreeClassifier():
    def __init__(self,
        n_bins=10, min_leaf=5, max_depth=3, n_samples=1.0, n_features=1.0, boot_replace=False, random_state=42,
        criterion="entropy/auc", bias_method="avg/w_avg/xtr", compound_bias_method="avg/xtr", orthogo_coef=1.0
    ):
        self.is_fit = False
        self.n_bins = n_bins
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.criterion = criterion
        self.n_features = n_features
        self.bias_method = bias_method
        self.orthogo_coef = orthogo_coef
        self.boot_replace = boot_replace
        self.random_state = random_state        
        self.compound_bias_method = compound_bias_method
        
        seed(random_state)
        np.random.seed(random_state)
        if (self.criterion != "entropy") and (self.criterion != "auc"):
            self.criterion = "auc"
            warnings.warn("criterion undefined -> setting criterion to auc")
        if (self.bias_method != "avg") and (self.bias_method != "w_avg") and (self.bias_method != "xtr"):
            self.bias_method = "avg"
            warnings.warn("bias_method undefined -> setting bias_method to avg")
        if (self.compound_bias_method != "avg") and (self.compound_bias_method != "xtr"):
            self.compound_bias_method = "avg"
            warnings.warn("compound_bias_method undefined -> setting compound_bias_method to avg")
            
        if "int" not in str(type(self.n_bins)):
            raise Exception("n_bins must be an int, not " + str(type(self.n_bins)))
        if "int" not in str(type(self.min_leaf)):
            raise Exception("min_leaf must be an int, not " + str(type(self.min_leaf)))
        if "int" not in str(type(self.max_depth)):
            raise Exception("max_depth must be an int, not " + str(type(self.max_depth)))
        if ("int" not in str(type(self.n_samples))) and ("float" not in str(type(self.n_samples))):
            raise Exception("n_samples must be an int or float, not " + str(type(self.n_samples)))
        if ("int" not in str(type(self.n_features))) and ("float" not in str(type(self.n_features))):
            raise Exception("n_features must be an int or float, not " + str(type(self.n_features)))
        if ("int" not in str(type(self.orthogo_coef))) and ("float" not in str(type(self.orthogo_coef))):
            raise Exception("orthogo_coef must be an int or float, not " + str(type(self.orthogo_coef)))
                                
    def fit(self, X="X", y="y", b="bias"):
        """
        X -> any_dim pandas.df or np.array: only floats
        y -> one_dim pandas.df or np.array: only binary
        b -> any_dim pandas.df or np.array: treated as str
        """
        seed(self.random_state)
        np.random.seed(self.random_state)
        self.X = np.array(X).astype(float)
        self.y = np.array(y).astype(str)
        self.b = np.array(b).astype(str)
        
        if (self.X.shape[0]!=self.y.shape[0]) or (self.X.shape[0]!=self.b.shape[0]) or (self.y.shape[0]!=self.b.shape[0]):
            raise Exception("X, y, and b lenghts do not match")    
        if len(self.y.shape)==1 or len(self.y.ravel())==len(self.X):
            self.y = self.y.ravel()
        if len(self.b.shape)==1 or len(self.b.ravel())==len(self.X):
            self.b = self.b.ravel()    
        
        
        self.y_neg, self.y_pos = np.unique(self.y)
        all_indexs = range(X.shape[0])
        all_features = range(X.shape[1])
        
        # self.samples -> set of indexs according to sampling
        if "int" in str(type(self.n_samples)):
            self.samples = np.array(sorted(
                np.random.choice(
                    all_indexs,
                    size=self.n_samples,
                    replace=self.boot_replace
                )
            ))
        else:
            self.samples = np.array(sorted(
                np.random.choice(
                    all_indexs,
                    size=int(self.n_samples*len(all_indexs)),
                    replace=self.boot_replace,
                )
            ))
        
        self.pred_th = sum(self.y[self.samples]==self.y_pos) / len(self.samples)
        
        # self.features -> set of features according to sampling
        if "int" in str(type(self.n_features)):
            self.features = sorted(
                np.random.choice(
                    all_features,
                    size=max(1, self.n_features),
                    replace=False
                )
            )
        else:
            self.features = sorted(
                np.random.choice(
                    all_features,
                    size=max(1, int(self.n_features*len(all_features))),
                    replace=False,
                )
            )
        
        # returns a dictionary as {feature: cutoff_candidate_i} meant as <=
        def get_candidate_splits(indexs):
            n_bins = self.n_bins
            candidate_splits = {}
            for feature in self.features:
                n_unique = len(np.unique(self.X[indexs,feature])) 
                if (n_unique-1) < n_bins:
                    candidate_splits[feature] = sorted(np.unique(self.X[indexs,feature]))[:-1]
                else:
                    lo = 1/n_bins
                    hi = lo * (n_bins-1)
                    quantiles = np.linspace(lo, hi, n_bins-1)
                    candidate_splits[feature] = list(np.quantile(self.X[indexs,feature], q=quantiles))
            return candidate_splits

        # return score of split (dependant on criterion) ONLY AUC implemented so far
        def evaluate_split(feature, split_value, indexs):
            
            # get auc of y associatated with split
            def get_auc_y(index_left, indexs):
                true_pos = sum(self.y[index_left]==self.y_pos)
                false_pos = sum(self.y[index_left]==self.y_neg)
                actual_pos = sum(self.y[indexs]==self.y_pos)
                actual_neg = sum(self.y[indexs]==self.y_neg)
                tpr = true_pos / actual_pos
                fpr = false_pos / actual_neg
                auc_y = (1 + tpr - fpr) / 2
                auc_y = max(1 - auc_y, auc_y)
                if np.isnan(auc_y):
                    auc_y = 1
                return auc_y
            
            # get auc of b associatated with split
            def get_auc_b(index_left, indexs):
                if len(self.b.shape)==1: #if we have only 1 bias column
                    b_unique = np.unique(self.b[indexs])
                    
                    if len(b_unique)==1: #if these indexs only contain 1 bias_value
                        auc_b = 1
                        
                    elif len(b_unique)==2: # if we are dealing with a binary case
                        true_pos = sum(self.b[index_left]==b_unique[0])
                        false_pos = sum(self.b[index_left]==b_unique[1])
                        actual_pos = sum(self.b[indexs]==b_unique[0])
                        actual_neg = sum(self.b[indexs]==b_unique[1])
                        tpr = true_pos / actual_pos
                        fpr = false_pos / actual_neg
                        auc_b = (1 + tpr - fpr) / 2
                        auc_b = max(1 - auc_b, auc_b)
                        if np.isnan(auc_b):
                            auc_b = 1
                        
                    else: # apply OvR
                        auc_storage = []
                        wts_storage = []
                        for b_uni in b_unique:
                            true_pos = sum(self.b[index_left]==b_uni)
                            false_pos = sum(self.b[index_left]!=b_uni)
                            actual_pos = sum(self.b[indexs]==b_uni)
                            actual_neg = sum(self.b[indexs]!=b_uni)
                            tpr = true_pos / actual_pos
                            fpr = false_pos / actual_neg
                            auc_b_uni = (1 + tpr - fpr) / 2
                            auc_b_uni = max(1 - auc_b_uni, auc_b_uni)
                            if np.isnan(auc_b_uni):
                                auc_b_uni = 1
                            auc_storage.append(auc_b_uni)
                            wts_storage.append(actual_pos)
                        if self.bias_method=="avg":
                            auc_b = np.mean(auc_storage)
                        elif self.bias_method=="w_avg":
                            auc_b = np.average(auc_storage, weights=wts_storage)
                        elif self.bias_method=="xtr":
                            auc_b = max(auc_storage)
                            
                # if we have more than 1 bias column
                else:
                    auc_b_columns = []
                    for b_column in range(self.b.shape[1]):
                        b_unique = np.unique(self.b[indexs, b_column])
                    
                        if len(b_unique)==1: #if these indexs only contain 1 bias_value
                            auc_b = 1

                        elif len(b_unique)==2: # if we are dealing with a binary case
                            true_pos = sum(self.b[index_left, b_column]==b_unique[0])
                            false_pos = sum(self.b[index_left, b_column]==b_unique[1])
                            actual_pos = sum(self.b[indexs, b_column]==b_unique[0])
                            actual_neg = sum(self.b[indexs, b_column]==b_unique[1])
                            tpr = true_pos / actual_pos
                            fpr = false_pos / actual_neg
                            auc_b = (1 + tpr - fpr) / 2
                            auc_b = max(1 - auc_b, auc_b)
                            if np.isnan(auc_b):
                                auc_b = 1
                                
                        else: # apply OvR
                            auc_storage = []
                            wts_storage = []
                            for b_uni in b_unique:
                                true_pos = sum(self.b[index_left, b_column]==b_uni)
                                false_pos = sum(self.b[index_left, b_column]!=b_uni)
                                actual_pos = sum(self.b[indexs, b_column]==b_uni)
                                actual_neg = sum(self.b[indexs, b_column]!=b_uni)
                                tpr = true_pos / actual_pos
                                fpr = false_pos / actual_neg
                                auc_b_uni = (1 + tpr - fpr) / 2
                                auc_b_uni = max(1 - auc_b_uni, auc_b_uni)
                                if np.isnan(auc_b_uni):
                                    auc_b_uni = 1
                                auc_storage.append(auc_b_uni)
                                wts_storage.append(actual_pos)
                            if self.bias_method=="avg":
                                auc_b = np.mean(auc_storage)
                            elif self.bias_method=="w_avg":
                                auc_b = np.average(auc_storage, weights=wts_storage)
                            elif self.bias_method=="xtr":
                                auc_b = max(auc_storage)
                        auc_b_columns.append(auc_b)
                    if self.compound_bias_method=="avg":
                        auc_b = np.mean(auc_b_columns)
                    elif self.compound_bias_method=="xtr":
                        auc_b = max(auc_b_columns)
                return auc_b
            
            index_left = indexs[self.X[indexs, feature] <= split_value]
            index_right = indexs[self.X[indexs, feature] > split_value]
            if self.criterion == "auc":
                auc_y = get_auc_y(index_left, indexs)
                auc_b = get_auc_b(index_left, indexs)
                score = auc_y - (self.orthogo_coef*auc_b)
            return score
                
        # return best (sscore, feature, split_value) dependant on criterion and indexs
        def get_best_split(indexs):
            best_score = -np.inf
            candidate_splits = get_candidate_splits(indexs)
            for feature in candidate_splits:
                for split_value in candidate_splits[feature]:
                    score = evaluate_split(feature, split_value, indexs)
                    if score >= best_score:
                        best_score = score
                        best_feature = feature
                        best_split_value = split_value
            
            if best_score==-np.inf: # this only happens if there are no more different feature values for splitting
                best_feature, best_split_value = np.nan, np.nan
            return best_score, best_feature, best_split_value
        
        # recursively grow the actual tree ---> {split1: {...}}
        def build_tree(indexs, step=0, old_score=-np.inf, new_score=-np.inf):
            ##print(indexs)
            tree={}
            if (                
                len(np.unique(self.y[indexs]))==1 or ( # no need to split if there is alreadyd only 1 y class
                len(indexs)<=self.min_leaf) or ( # minimum number to consider a node as a leaf
                #new_score<old_score) or ( # if score is lower after split
                step==self.max_depth) # if we've reached the max depth in the tree
            ):
                return indexs

            else:
                step += 1
                score, feature, split_value = get_best_split(indexs)
                old_score = copy(new_score)
                new_score = copy(score)
                if new_score==-np.inf: ## in case no more feature values exist for splitting
                    return indexs
                
                ##print(indexs)
                left_indexs = indexs[self.X[indexs, feature]<=split_value]
                right_indexs = indexs[self.X[indexs, feature]>split_value]
                
                if (len(left_indexs)==0) or (len(right_indexs)==0):
                    return indexs
                
                else:
                    tree[(feature, split_value)] = {
                        "<=": build_tree(left_indexs, step=copy(step), old_score=copy(old_score), new_score=copy(new_score)),
                        ">":  build_tree(right_indexs, step=copy(step), old_score=copy(old_score), new_score=copy(new_score))
                    }

            return tree
        
        self.tree = build_tree(self.samples)
        self.is_fit=True   
    
    def predict(self, X):
        if self.is_fit:
            X = np.array(X).astype(float)
            if len(X.shape)!=2:
                raise Exception("X.shape must be (n_instances, features)")
            predictions = []
            for x in X:
                sub_tree = self.tree

                while type(sub_tree) != type(np.array([1])):

                    feature, value = list(sub_tree.keys())[0]
                    if x[feature] <= value:
                        sub_tree = sub_tree[feature, value]["<="]

                    else:
                        sub_tree = sub_tree[feature, value][">"]

                prediction = sum(self.y[sub_tree]==self.y_pos) / len(sub_tree)
                if prediction >= self.pred_th:
                    predictions.append(self.y_pos)
                else:
                    predictions.append(self.y_neg)

            return np.array(predictions)
        else:
            raise Exception("tree has not been fit")
            
    def predict_proba(self, X):
        if self.is_fit:
            X = np.array(X).astype(float)
            if len(X.shape)!=2:
                raise Exception("X.shape must be (n_instances, features)")
            predictions = []
            for x in X:
                sub_tree = self.tree
                while type(sub_tree) != type(np.array([1])):

                    feature, value = list(sub_tree.keys())[0]
                    if x[feature] <= value:
                        sub_tree = sub_tree[feature, value]["<="]

                    else:
                        sub_tree = sub_tree[feature, value][">"]

                prediction = sum(self.y[sub_tree]==self.y_pos) / len(sub_tree)
                predictions.append(prediction)
         
            return np.array(predictions)
        else:
            raise Exception("tree has not been fit")
            
    
    def __str__(self):
        string = "BiasConstraintDecisionTreeClassifier():" + "\n" + \
                "  is_fit=" + str(self.is_fit) + "\n" + \
                "  n_bins=" + str(self.n_bins) + "\n" + \
                "  min_leaf=" + str(self.min_leaf) + "\n" + \
                "  max_depth=" + str(self.max_depth) + "\n" + \
                "  n_samples=" + str(self.n_samples) + "\n" + \
                "  criterion=" + str(self.criterion) + "\n" + \
                "  n_features=" + str(self.n_features) + "\n" + \
                "  bias_method=" +str(self.bias_method) + "\n" + \
                "  orthogo_coef=" +str(self.orthogo_coef) + "\n" + \
                "  boot_replace=" +str(self.boot_replace) + "\n" + \
                "  random_state=" + str(self.random_state) + "\n" + \
                "  compound_bias_method=" + str(self.compound_bias_method)
        return string

    def __repr__(self):
        string = "BiasConstraintDecisionTreeClassifier():" + "\n" + \
                "  is_fit=" + str(self.is_fit) + "\n" + \
                "  n_bins=" + str(self.n_bins) + "\n" + \
                "  min_leaf=" + str(self.min_leaf) + "\n" + \
                "  max_depth=" + str(self.max_depth) + "\n" + \
                "  n_samples=" + str(self.n_samples) + "\n" + \
                "  criterion=" + str(self.criterion) + "\n" + \
                "  n_features=" + str(self.n_features) + "\n" + \
                "  bias_method=" +str(self.bias_method) + "\n" + \
                "  orthogo_coef=" +str(self.orthogo_coef) + "\n" + \
                "  boot_replace=" +str(self.boot_replace) + "\n" + \
                "  random_state=" + str(self.random_state) + "\n" + \
                "  compound_bias_method=" + str(self.compound_bias_method)
        return string
    
class BiasConstraintRandomForestClassifier():
    def __init__(self, n_estimators=500, n_jobs=-1,
        n_bins=10, min_leaf=5, max_depth=3, n_samples=1.0, n_features=1.0, boot_replace=True, random_state=42,
        criterion="auc", bias_method="avg", compound_bias_method="avg", orthogo_coef=1.0
    ):
        """
        Bias Constraint Forest Classifier
        n_estimators -> int: BCDTress to generate
        n_bins -> int: feature quantiles to evaluate at each split
        min_leaf -> int: largest number of samples for a node to stop splitting
        max_depth -> int: max number of allowed splits per tree
        n_samples -> int: number of samples to bootstrap
                  -> float: proportion of samples to bootstrap
        n_features -> int: number of samples to bootstrap
                   -> float: proportion of samples to bootstrap
        boot_replace -> bool: bootstrap strategy (with out without replacement)
        random_state -> int: seed for all random processes
        criterion -> str: ["entropy", "auc"] score criterion for splitting
        bias_method -> str: ["avg", "w_avg", "xtr"] OvR approach for categorical bias attribute
        compound_bias_method -> str: ["avg", "xtr"] aggregation approach for multiple bias attributes
        orthogo_coef -> int/float: strength of bias constraint
        n_jobs -> int: CPU usage / -1 for all 
        """
        self.is_fit = False
        self.n_bins = n_bins
        self.n_jobs = n_jobs
        self.min_leaf = min_leaf
        self.max_depth = max_depth
        self.n_samples = n_samples
        self.criterion = criterion
        self.n_features = n_features
        self.bias_method = bias_method
        self.orthogo_coef = orthogo_coef
        self.boot_replace = boot_replace
        self.random_state = random_state        
        self.n_estimators = n_estimators
        self.compound_bias_method = compound_bias_method
        
        seed(random_state)
        np.random.seed(random_state)
        if (self.criterion != "entropy") and (self.criterion != "auc"):
            self.criterion = "auc"
            warnings.warn("criterion undefined -> setting criterion to auc")
        if (self.bias_method != "avg") and (self.bias_method != "w_avg") and (self.bias_method != "xtr"):
            self.bias_method = "avg"
            warnings.warn("bias_method undefined -> setting bias_method to avg")
        if (self.compound_bias_method != "avg") and (self.compound_bias_method != "xtr"):
            self.compound_bias_method = "avg"
            warnings.warn("compound_bias_method undefined -> setting compound_bias_method to avg")
            
        if "int" not in str(type(self.n_bins)):
            raise Exception("n_bins must be an int, not " + str(type(self.n_bins)))
        if "int" not in str(type(self.min_leaf)):
            raise Exception("min_leaf must be an int, not " + str(type(self.min_leaf)))
        if "int" not in str(type(self.max_depth)):
            raise Exception("max_depth must be an int, not " + str(type(self.max_depth)))
        if ("int" not in str(type(self.n_samples))) and ("float" not in str(type(self.n_samples))):
            raise Exception("n_samples must be an int or float, not " + str(type(self.n_samples)))
        if ("int" not in str(type(self.n_features))) and ("float" not in str(type(self.n_features))):
            raise Exception("n_features must be an int or float, not " + str(type(self.n_features)))
        if ("int" not in str(type(self.orthogo_coef))) and ("float" not in str(type(self.orthogo_coef))):
            raise Exception("orthogo_coef must be an int or float, not " + str(type(self.orthogo_coef)))
        
        
        # Generating BCDForest
        dts = [
            BiasConstraintDecisionTreeClassifier(
                n_bins=copy(self.n_bins),
                min_leaf=copy(self.min_leaf),
                max_depth=copy(self.max_depth),
                n_samples=copy(self.n_samples),
                criterion=copy(self.criterion),
                random_state=copy(self.random_state+i),
                n_features=copy(self.n_features),
                bias_method=copy(self.bias_method),
                orthogo_coef=copy(self.orthogo_coef),
                boot_replace=copy(self.boot_replace),
                compound_bias_method=copy(self.compound_bias_method),
            )
            for i in range(self.n_estimators)
        ]
        self.trees = dts
        
    def fit(self, X, y, s):
        def batch(iterable, n_jobs=1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count() # - 1 # -1 so that our laptop doesn't freeze
            l = len(iterable)
            n = ceil(l / n_jobs)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]
                
        def fit_trees_parallel(i, dt_batch, X ,y ,s):
            dt_batch = dt_batches[i]
            fit_dt_batch = []
            for dt in tqdm(dt_batch, desc=str(i)):
                dt.fit(X, y, s)
                fit_dt_batch.append(dt)
            return fit_dt_batch

        # Fitting
        dts = self.trees
        dt_batches = list(batch(dts, n_jobs=self.n_jobs))
        fit_dt_batches = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_trees_parallel)(
                copy(i),
                copy(dt_batches),
                copy(X),
                copy(y),
                copy(s)
            ) for i in (range(len(copy(dt_batches))))
        )
        fit_dts = [item for sublist in fit_dt_batches for item in sublist]
        self.trees = fit_dts
        self.fit = True
    
    def predict(self, X):
        def batch(iterable, n_jobs=1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count() # - 1 # -1 so that our laptop doesn't freeze
            l = len(iterable)
            n = ceil(l / n_jobs)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]
        def predict_proba_parallel(i, fit_dt_batch, X):
            fit_dt_batch = fit_dt_batch[i]
            y_preds = []
            for dt in tqdm(fit_dt_batch, desc=str(i)):
                y_preds.append(dt.predict_proba(X))
            return np.array(y_preds)
        
        if not self.fit:
            warnings.warn("Forest has not been fit(X,y,s)")
        else:
            # Predicting
            fit_dts = self.trees
            fit_dt_batches = list(batch(fit_dts, n_jobs=self.n_jobs))
            y_preds = Parallel(n_jobs=self.n_jobs)(
                delayed(predict_proba_parallel)(
                    copy(i),
                    copy(fit_dt_batches),
                    copy(X),
                ) for i in (range(len(copy(fit_dt_batches))))
            )
            y_pred = np.mean(np.array([item for sublist in y_preds for item in sublist]), axis=0)
            return y_pred

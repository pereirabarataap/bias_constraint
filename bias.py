import os

# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1" 
# os.environ["OMP_NUM_THREADS"] = "1" 

import dccp
import random
import pprint
import warnings
import operator
import functools
import cvxpy as cp
import numpy as np
import pandas as pd
import seaborn as sb
from tqdm import tqdm
from math import ceil
import multiprocessing
from random import seed
import matplotlib.pyplot as plt
from dccp.problem import is_dccp
from copy import deepcopy as copy
from scipy.optimize import minimize
from joblib import delayed, Parallel
from scipy.stats import mode, entropy
from tqdm.notebook import tqdm as tqdm_n
from scipy.special import expit as sigmoid
from collections import Counter, defaultdict
from sklearn.model_selection import GroupKFold as GKF
from sklearn.preprocessing import PolynomialFeatures as PF
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler as SS, RobustScaler as RS

# os.environ["MKL_NUM_THREADS"] = "1" 
# os.environ["NUMEXPR_NUM_THREADS"] = "1" 
# os.environ["OMP_NUM_THREADS"] = "1" 

class DensityRatioEstimator:
    """
    Class to accomplish direct density estimation implementing the original KLIEP 
    algorithm from Direct Importance Estimation with Model Selection
    and Its Application to Covariate Shift Adaptation by Sugiyama et al. 
    
    The training set is distributed via 
                                            train ~ p(x)
    and the test set is distributed via 
                                            test ~ q(x).
                                            
    The KLIEP algorithm and its variants approximate w(x) = q(x) / p(x) directly. The predict function returns the
    estimate of w(x). The function w(x) can serve as sample weights for the training set during
    training to modify the expectation function that the model's loss function is optimized via,
    i.e.
    
            E_{x ~ w(x)p(x)} loss(x) = E_{x ~ q(x)} loss(x).
    
    Usage : 
        The fit method is used to run the KLIEP algorithm using LCV and returns value of J 
        trained on the entire training/test set with the best sigma found. 
        Use the predict method on the training set to determine the sample weights from the KLIEP algorithm.
    """
    
    def __init__(self, max_iter=5000, num_params=[.5], epsilon=1e-4, cv=5, sigmas=[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3], random_state=None, verbose=0):
        """ 
        Direct density estimation using an inner LCV loop to estimate the proper model. Can be used with sklearn
        cross validation methods with or without storing the inner CV. To use a standard grid search.
        
        
        max_iter : Number of iterations to perform
        num_params : List of number of test set vectors used to construct the approximation for inner LCV.
                     Must be a float. Original paper used 10%, i.e. =.1
        sigmas : List of sigmas to be used in inner LCV loop.
        epsilon : Additive factor in the iterative algorithm for numerical stability.
        """
        self.max_iter = max_iter
        self.num_params = num_params
        self.epsilon = epsilon
        self.verbose = verbose
        self.sigmas = sigmas
        self.cv = cv
        self.random_state = 0
        
    def fit(self, X_train, X_test, alpha_0=None):
        """ Uses cross validation to select sigma as in the original paper (LCV).
            In a break from sklearn convention, y=X_test.
            The parameter cv corresponds to R in the original paper.
            Once found, the best sigma is used to train on the full set."""
        
        # LCV loop, shuffle a copy in place for performance.
        cv = self.cv
        chunk = int(X_test.shape[0]/float(cv))
        if self.random_state is not None:
            np.random.seed(self.random_state)
        X_test_shuffled = X_test.copy()
        np.random.shuffle(X_test_shuffled)
        
        j_scores = {}
        
        if type(self.sigmas) != list:
            self.sigmas = [self.sigmas]
        
        if type(self.num_params) != list:
            self.num_params = [self.num_params]
        
        if len(self.sigmas) * len(self.num_params) > 1:
            # Inner LCV loop
            for num_param in self.num_params:
                for sigma in self.sigmas:
                    j_scores[(num_param,sigma)] = np.zeros(cv)
                    for k in range(1,cv+1):
                        if self.verbose > 0:
                            print('Training: sigma: %s    R: %s' % (sigma, k))
                        X_test_fold = X_test_shuffled[(k-1)*chunk:k*chunk,:] 
                        j_scores[(num_param,sigma)][k-1] = self._fit(X_train=X_train, 
                                                         X_test=X_test_fold,
                                                         num_parameters = num_param,
                                                         sigma=sigma)
                    j_scores[(num_param,sigma)] = np.mean(j_scores[(num_param,sigma)])

            sorted_scores = sorted([x for x in j_scores.items() if np.isfinite(x[1])], key=lambda x :x[1], reverse=True)
            if len(sorted_scores) == 0:
                warnings.warn('LCV failed to converge for all values of sigma.')
                return self
            self._sigma = sorted_scores[0][0][1]
            self._num_parameters = sorted_scores[0][0][0]
            self._j_scores = sorted_scores
        else:
            self._sigma = self.sigmas[0]
            self._num_parameters = self.num_params[0]
            # best sigma
        self._j = self._fit(X_train=X_train, X_test=X_test_shuffled, num_parameters=self._num_parameters, sigma=self._sigma)

        return self # Compatibility with sklearn
        
    def _fit(self, X_train, X_test, num_parameters, sigma, alpha_0=None):
        """ Fits the estimator with the given parameters w-hat and returns J"""
        
        num_parameters = num_parameters
        
        if type(num_parameters) == float:
            num_parameters = int(X_test.shape[0] * num_parameters)

        self._select_param_vectors(X_test=X_test, 
                                   sigma=sigma,
                                   num_parameters=num_parameters)
        
        X_train = self._reshape_X(X_train)
        X_test = self._reshape_X(X_test)
        
        if alpha_0 is None:
            alpha_0 = np.ones(shape=(num_parameters,1))/float(num_parameters)
        
        self._find_alpha(X_train=X_train,
                         X_test=X_test,
                         num_parameters=num_parameters,
                         epsilon=self.epsilon,
                         alpha_0 = alpha_0,
                         sigma=sigma)
        
        return self._calculate_j(X_test,sigma=sigma)
    
    def _calculate_j(self, X_test, sigma):
        return np.log(self.predict(X_test,sigma=sigma)).sum()/X_test.shape[0]
    
    def score(self, X_test):
        """ Return the J score, similar to sklearn's API """
        return self._calculate_j(X_test=X_test, sigma=self._sigma)

    @staticmethod   
    def _reshape_X(X):
        """ Reshape input from mxn to mx1xn to take advantage of numpy broadcasting. """
        if len(X.shape) != 3:
            return X.reshape((X.shape[0],1,X.shape[1]))
        return X
    
    def _select_param_vectors(self, X_test, sigma, num_parameters):
        """ X_test is the test set. b is the number of parameters. """ 
        indices = np.random.choice(X_test.shape[0], size=num_parameters, replace=False)
        self._test_vectors = X_test[indices,:].copy()
        self._phi_fitted = True
        
    def _phi(self, X, sigma=None):
        
        if sigma is None:
            sigma = self._sigma
        
        if self._phi_fitted:
            return np.exp(-np.sum((X-self._test_vectors)**2, axis=-1)/(2*sigma**2))
        raise Exception('Phi not fitted.')

    def _find_alpha(self, alpha_0, X_train, X_test, num_parameters, sigma, epsilon):
        A = np.zeros(shape=(X_test.shape[0],num_parameters))
        b = np.zeros(shape=(num_parameters,1))

        A = self._phi(X_test, sigma)
        b = self._phi(X_train, sigma).sum(axis=0) / X_train.shape[0] 
        b = b.reshape((num_parameters, 1))
        
        out = alpha_0.copy()
        for k in range(self.max_iter):
            out += epsilon*np.dot(np.transpose(A),1./np.dot(A,out))
            out += b*(((1-np.dot(np.transpose(b),out))/np.dot(np.transpose(b),b)))
            out = np.maximum(0,out)
            out /= (np.dot(np.transpose(b),out))
            
        self._alpha = out
        self._fitted = True
        
    def predict(self, X, sigma=None):
        """ Equivalent of w(X) from the original paper."""
        
        X = self._reshape_X(X)
        if not self._fitted:
            raise Exception('Not fitted!')
        return np.dot(self._phi(X, sigma=sigma), self._alpha).reshape((X.shape[0],))

class StratifiedGroupKFold:
    
    def __init__(self, n_splits=5, random_state=42):
        
        self.k=n_splits
        self.seed=random_state
    
    def split(self, X, y, s):
        k = self.k
        seed = self.seed
        groups = s
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)

        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(k):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(k):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices

class CovarianceConstraintLinearRegression:
    """
    Covariance-Constraint Logistic Regression
    """
    def __init__(self, **args):
        
        """
        base_covariance: {None, dict(str: float)}
            dictionary of sens-attr-val (key) to abs unconstrained-covariance-value
            unconstrained-covariance-value is the measured between prediciton
            and the binarised sens-attr-value vector from a normal Regression
            
            covariance(prediciton, bin-sens-attr-value) = 1/n * sum (prediciton * z),
            where z = bin-sens-attr-value - bin-sens-attr-value.mean()
                
            if None, computes the dictionary from additional  unconstrained fitting.
            * default: None
        
        cov_trehshold: {False, float, dict(str: float)}
            the constrained covariance value to apply during fitting
            if float, apply the same threshold to all sens-attr-values
            if dict, apply value-specific threshold
            if False, apply cov_coefficient instead
            * default: False
            
        cov_coefficient: {False, float, dict(str: float)}
            the coefficient value to apply to the base_covariance during fitting
            if float [0,1] -> apply the same coefficient to all sens-attr-values
            if dict, apply value-specific coefficient
            if False, apply cov_trehshold instead
            * default: 1e-1
            
        add_intercept: bool
            if the classifier should add a 1s column to serve as intercept
            * default: True
        """
        
        keys = ["base_covariance", "add_intercept", "cov_trehshold", "cov_coefficient"]
        args_keys = list(args.keys())
        
        if "base_covarianve" not in args_keys:
            #self.base_covariance = None
            args["base_covariance"] = None
        
        if "add_intercept" not in args_keys:
            #self.add_intercept = True
            args["add_intercept"] = True
            
        if "cov_trehshold" not in args_keys:
            #self.cov_trehshold = False
            args["cov_trehshold"] = False
            
        if "cov_coefficient" not in args_keys:
            #self.cov_coefficient = 1e-1
            args["cov_coefficient"] = 1e-1
    
        self.args = args
        self.is_fit = False
    
    def fit(self, X, y, s, weights=None):
        """
        X: np.array.astype(float) shape(n,m)
        y: np.array.astype(int)   shape(n,)
        s: np.array.astype(str)   shape(n,)
        """
       
        if type(weights) == type(None):
            weights = np.ones_like(y)
        X = np.array(X).astype(float)
        y = np.array(y).astype(float)
        s = pd.get_dummies(np.array(s).astype(str))
        z = s - s.mean()
        
        if self.args["add_intercept"] == True:
            X = np.concatenate((X, np.ones(shape=(X.shape[0],1))), axis=1)

        if self.args["base_covariance"] == None:
            base_covariance = {}
            w = cp.Variable(X.shape[1])
    
            loss = cp.sum_squares(X @ w - y) / X.shape[0]
            obj = cp.Minimize(loss)
            prob = cp.Problem(obj)
            prob.solve()

            pred = X @ w.value
            for attr_value in z.columns:
                base_covariance[attr_value] = abs(sum((pred) * z[attr_value]) / X.shape[0])
            self.args["base_covariance"] = base_covariance
        
        # make constraints sens-atr-value specific
        constraints = []
        w = cp.Variable(X.shape[1])
        if self.args["cov_trehshold"] == False:
            cov_coefficient = self.args["cov_coefficient"]
            base_covariance = self.args["base_covariance"]
            if type(cov_coefficient) == type({}):
                for attr_value in z.columns:
                    cov_coef = cov_coefficient[attr_value]
                    cov_trehshold = base_covariance[attr_value] * cov_coef
                    covariance = cp.sum(cp.multiply(z[attr_value], X @ w)) / X.shape[0]
                    constraints.append(covariance >= -cov_trehshold)
                    constraints.append(covariance <= cov_trehshold)
            else:
                if cov_coefficient < 1: # if not, then we don't need constraints
                    for attr_value in z.columns:
                        cov_trehshold = base_covariance[attr_value] * cov_coefficient
                        covariance = cp.sum(cp.multiply(z[attr_value], X @ w)) / X.shape[0]
                        constraints.append(covariance >= -cov_trehshold)
                        constraints.append(covariance <= cov_trehshold)
        
        loss = cp.sum_squares(X @ w - y) / X.shape[0]
        
        obj = cp.Minimize(loss)
        prob = cp.Problem(obj, constraints=constraints)
        fun_value = prob.solve()
        
        self.is_fit = True
        self.coefs = w.value
        self.fun_value = fun_value
        
    def predict(self, X):
        if self.args["add_intercept"] == True:
            X = np.concatenate((X, np.ones(shape=(X.shape[0],1))), axis=1)
            pred = X @ self.coefs
        else:
            pred = X @ self.coefs
        return pred
    
    def __str__(self):
        args_str = pprint.pformat(self.args, indent=4)
        return("Covariance-Constraint Logistic Regression\n" + args_str)
    
    def __repr__(self):
        args_str = pprint.pformat(self.args, indent=4)
        return("Covariance-Constraint Logistic Regression\n" + args_str)

class CovarianceConstraintLogisticRegression:
    """
    Covariance-Constraint Logistic Regression
    """
    def __init__(self, **args):
        
        """
        base_covariance: {None, dict(str: float)}
            dictionary of sens-attr-val (key) to abs unconstrained-covariance-value
            unconstrained-covariance-value is the measured between logitraw prediciton
            and the binarised sens-attr-value vector from a normal Logistic Regression
            
            covariance(logitraw, bin-sens-attr-value) = 1/n * sum (logitraw * z),
            where z = bin-sens-attr-value - bin-sens-attr-value.mean()
                
            if None, computes the dictionary from additional  unconstrained fitting.
            * default: None
        
        cov_trehshold: {False, float, dict(str: float)}
            the constrained covariance value to apply during fitting
            if float, apply the same threshold to all sens-attr-values
            if dict, apply value-specific threshold
            if False, apply cov_coefficient instead
            * default: False
            
        cov_coefficient: {False, float, dict(str: float)}
            the coefficient value to apply to the base_covariance during fitting
            if float [0,1] -> apply the same coefficient to all sens-attr-values
            if dict, apply value-specific coefficient
            if False, apply cov_trehshold instead
            * default: 1e-1
            
        add_intercept: bool
            if the classifier should add a 1s column to serve as intercept
            * default: True
        """
        
        keys = ["base_covariance", "add_intercept", "cov_trehshold", "cov_coefficient"]
        args_keys = list(args.keys())
        
        if "base_covarianve" not in args_keys:
            #self.base_covariance = None
            args["base_covariance"] = None
        
        if "add_intercept" not in args_keys:
            #self.add_intercept = True
            args["add_intercept"] = True
            
        if "cov_trehshold" not in args_keys:
            #self.cov_trehshold = False
            args["cov_trehshold"] = False
            
        if "cov_coefficient" not in args_keys:
            #self.cov_coefficient = 1e-1
            args["cov_coefficient"] = 1e-1
    
        self.args = args
        self.is_fit = False
    
    def fit(self, X, y, s, weights=None):
        """
        X: np.array.astype(float) shape(n,m)
        y: np.array.astype(int)   shape(n,)
        s: np.array.astype(str)   shape(n,)
        """
       
        if type(weights) == type(None):
            weights = np.ones_like(y)
        X = np.array(X).astype(float)
        y = np.array(y).astype(int)
        s = pd.get_dummies(np.array(s).astype(str))
        z = s - s.mean()
        
        if self.args["add_intercept"] == True:
            X = np.concatenate((X, np.ones(shape=(X.shape[0],1))), axis=1)

        if self.args["base_covariance"] == None:
            base_covariance = {}
            w = cp.Variable(X.shape[1])
    
            loss = -1 * cp.sum(
                cp.multiply(
                    weights,
                    cp.multiply(y, X @ w) - cp.logistic(X @ w)
                )
            ) / X.shape[0]
            obj = cp.Minimize(loss)
            prob = cp.Problem(obj)
            prob.solve(
                abstol=1e-2,
                reltol=1e-2,
                feastol=1e-2,                
                abstol_inacc=1e-2,
                reltol_inacc=1e-2,
                feastol_inacc=1e-2,                
                max_iters=int(1e2)
            )
            pred = X @ w.value
            for attr_value in z.columns:
                base_covariance[attr_value] = abs(sum(pred * z[attr_value]) / X.shape[0])
            self.args["base_covariance"] = base_covariance
        
        # make constraints sens-atr-value specific
        constraints = []
        w = cp.Variable(X.shape[1])
        if self.args["cov_trehshold"] == False:
            cov_coefficient = self.args["cov_coefficient"]
            base_covariance = self.args["base_covariance"]
            if type(cov_coefficient) == type({}):
                for attr_value in z.columns:
                    cov_coef = cov_coefficient[attr_value]
                    cov_trehshold = base_covariance[attr_value] * cov_coef
                    covariance = cp.sum(cp.multiply(z[attr_value], X @ w)) / X.shape[0]
                    constraints.append(covariance >= -cov_trehshold)
                    constraints.append(covariance <= cov_trehshold)
            else:
                if cov_coefficient < 1: # if not, then we don't need constraints
                    for attr_value in z.columns:
                        cov_trehshold = base_covariance[attr_value] * cov_coefficient
                        covariance = cp.sum(cp.multiply(z[attr_value], X @ w)) / X.shape[0]
                        constraints.append(covariance >= -cov_trehshold)
                        constraints.append(covariance <= cov_trehshold)
        
        loss = -1 * cp.sum(
            cp.multiply(
                weights,
                cp.multiply(y, X @ w) - cp.logistic(X @ w)
            )
        ) / X.shape[0]
        
        obj = cp.Minimize(loss)
        prob = cp.Problem(obj, constraints=constraints)
        fun_value = prob.solve(
            abstol=1e-2,
            reltol=1e-2,
            feastol=1e-2,                
            abstol_inacc=1e-2,
            reltol_inacc=1e-2,
            feastol_inacc=1e-2,                
            max_iters=int(1e2)
        )
        self.is_fit = True
        self.coefs = w.value
        self.fun_value = fun_value
        
    def predict_raw(self, X):
        if self.args["add_intercept"] == True:
            X = np.concatenate((X, np.ones(shape=(X.shape[0],1))), axis=1)
        pred = np.array([-1 * X @ self.coefs, X @ self.coefs]).T
        return pred
    
    def predict_proba(self, X):
        return sigmoid(self.predict_raw(X))
    
    def predict(self, X):
        return np.argmax(self.predict_raw(X), axis=1)
    
    def __str__(self):
        args_str = pprint.pformat(self.args, indent=4)
        return("Covariance-Constraint Logistic Regression\n" + args_str)
    
    def __repr__(self):
        args_str = pprint.pformat(self.args, indent=4)
        return("Covariance-Constraint Logistic Regression\n" + args_str)

class BiasConstraintLogisticRegression():
    
    def __init__(self, ortho=1, l1_reg_factor=0, add_intersect=True, ortho_method="avg", random_state=42):
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
        """
        
        self.ortho=ortho
        self.is_fit=False
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
            
            # corelation coef **2
            score = score.reshape(len(score), 1)
            if len(np.unique(score))==1:
                sens_reg = 1
            else:
                if ortho_method=="avg":
                    sens_reg = sum(
                        corr2_coeff(s.T,score.T).ravel()**2
                    ) / s.shape[1]
                elif ortho_method=="max":
                    sens_reg = max(
                        corr2_coeff(s.T,score.T).ravel()**2
                    )
                elif ortho_method=="w_avg":
                    sens_reg = np.average(
                        corr2_coeff(s.T,score.T).ravel()**2, weights=np.sum(s, axis=0)
                    )
                elif ortho_method=="inv_w_avg":
                    sens_reg = np.average(
                        corr2_coeff(s.T,score.T).ravel()**2, weights=len(s)/np.sum(s, axis=0)
                    )

            return loss + ortho*sens_reg
        
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
        
        message = "failure"
        while "success" not in message:
            result = minimize(
                fun=loss,
                x0=coefs,
                args=(X, y, s, self.ortho, self.add_intersect, self.ortho_method),
                method="SLSQP",
                jac="3-point",
                tol=1e-7,
            )
            message = result.message
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
            message = "failure"
            while "success" not in message:
                result = minimize(
                    fun=loss,
                    x0=coefs,
                    args=(X_retained, y, s, self.ortho, self.add_intersect, self.ortho_method),
                    method="SLSQP",
                    jac="3-point",
                    tol=1e75,
                )
                message = result.message
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

class BiasConstraintDecisionTreeClassifier():
    def __init__(self,
        n_bins=10, min_leaf=3, max_depth=5, n_samples=1.0, n_features="auto", boot_replace=True, random_state=42,
        criterion='{"entropy", "auc", "faht", "ig", "fg"}', bias_method="avg", compound_bias_method="avg", orthogo_coef=.6
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
        
    
        if self.criterion not in ["entropy", "auc", "faht", "ig", "fg", "kamiran_add", "kamiran_div", "kamiran_sub"]:
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
#         if ("int" not in str(type(self.n_features))) and ("float" not in str(type(self.n_features))):
#             raise Exception("n_features must be an int or float, not " + str(type(self.n_features)))
        if ("int" not in str(type(self.orthogo_coef))) and ("float" not in str(type(self.orthogo_coef))):
            raise Exception("orthogo_coef must be an int or float, not " + str(type(self.orthogo_coef)))
                                
    def fit(self, X="X", y="y", b="bias"):
        """
        X -> any_dim pandas.df or np.array: numerical/categorical
        y -> one_dim pandas.df or np.array: only binary
        b -> any_dim pandas.df or np.array: treated as str
        """
        np.random.seed(self.random_state)
        self.X = np.array(X)
        if self.criterion=="auc":
            self.y = np.array(y).astype(int)
            self.b = np.array(b).astype(int)
        else:
            self.y = np.array(y).astype(str)
            self.b = np.array(b).astype(str)

        if (self.X.shape[0]!=self.y.shape[0]) or (self.X.shape[0]!=self.b.shape[0]) or (self.y.shape[0]!=self.b.shape[0]):
            raise Exception("X, y, and b lenghts do not match")    
        if len(self.y.shape)==1 or len(self.y.ravel())==len(self.X):
            self.y = self.y.ravel()
        if len(self.b.shape)==1 or len(self.b.ravel())==len(self.X):
            self.b = self.b.ravel()    
        
        self.b_neg, self.b_pos = np.unique(self.b)
        self.y_neg, self.y_pos = (0, 1)
        all_indexs = range(X.shape[0])
        all_features = range(X.shape[1])
        self.features = all_features
        # self.samples -> set of indexs according to sampling
        if "int" in str(type(self.n_samples)):
            self.samples = np.array(
                np.random.choice(
                    all_indexs,
                    size=self.n_samples,
                    replace=self.boot_replace
                )
            )
        else:
            self.samples = np.array(
                np.random.choice(
                    all_indexs,
                    size=int(self.n_samples*len(all_indexs)),
                    replace=self.boot_replace,
                )
            )
        
        self.pred_th = sum(self.y[self.samples]==self.y_pos) / len(self.samples)

        def choose_features():   
            if "int" in str(type(self.max_features)):
                chosen_features = np.random.choice(
                        features,
                        size=max(1, self.max_features),
                        replace=False
                )
            elif ("auto" in str(self.max_features)) or ("sqrt" in str(self.max_features)):
                chosen_features = np.random.choice(
                        features,
                        size=max(1, int(np.sqrt(len(features)))),
                        replace=False
                )
            elif "log" in str(self.max_features):
                chosen_features = np.random.choice(
                        features,
                        size=max(1, int(np.log2(len(features)))),
                        replace=False
                )
            else:
                chosen_features = np.random.choice(
                        features,
                        size=max(1, int(self.max_features*len(features))),
                        replace=False,
                )
            return chosen_features
    
        # returns a dictionary as {feature: cutoff_candidate_i} meant as <
        def get_candidate_splits(indexs):
            
            n_bins = self.n_bins
            candidate_splits = {}
            chosen_features = choose_features()
            #print(chosen_features)
            for feature in chosen_features:
                if "str" in str(type(self.X[0,feature])):
                    candidate_splits[feature] = list(pd.value_counts(self.X[indexs, feature]).keys())
                else:
                    n_unique = len(np.unique(self.X[indexs,feature])) 
                    values = np.unique(self.X[indexs, feature])
                    n_unique = len(values)
                    if (n_unique) > self.n_bins:
                        lo = 1/self.n_bins
                        hi = lo * (self.n_bins-1)
                        quantiles = np.linspace(lo, hi, self.n_bins-1)
                        values = list(np.quantile(values, q=quantiles))
                    candidate_splits[feature] = values

            return candidate_splits

        # return score of split (dependant on criterion) ONLY AUC implemented so far
        def evaluate_split(feature, split_value, indexs):
            
            # get auc of y associatated with split
            def get_auc_y(index_left, index_right):
                
                n_left = len(index_left)
                n_right = len(index_right)
                y_left = self.y[index_left]
                y_right = self.y[index_right]
                proba_left = sum(y_left==1)/n_left
                proba_right = sum(y_right==1)/n_right
                
                y_prob = np.concatenate(
                    (np.repeat(proba_left, n_left), np.repeat(proba_right, n_right))
                )
                y_true = np.concatenate(
                    (y_left, y_right)
                )
            
                auc_y = roc_auc_score(y_true, y_prob)
                
                return auc_y
            
            # get auc of b associatated with split
            def get_auc_b(index_left, index_right):
                indexs = np.concatenate((index_left, index_right))
                if len(self.b.shape)==1: #if we have only 1 bias column
                    b_unique = np.unique(self.b[indexs])
                    
                    if len(b_unique)==1: #if these indexs only contain 1 bias_value
                        auc_b = 1
                        
                    elif len(b_unique)==2: # if we are dealing with a binary case
                        n_left = len(index_left)
                        n_right = len(index_right)
                        y_left = self.y[index_left]
                        y_right = self.y[index_right]
                        proba_left = sum(y_left==1)/n_left
                        proba_right = sum(y_right==1)/n_right

                        y_prob = np.concatenate(
                            (np.repeat(proba_left, n_left), np.repeat(proba_right, n_right))
                        )
                        y_true = np.concatenate(
                            (y_left, y_right)
                        )
                        
                        b_left = self.b[index_left]
                        b_right = self.b[index_right]
                        b_true = np.concatenate(
                            (b_left, b_right)
                        )
                        auc_b = roc_auc_score(b_true, y_prob)
                        auc_b = max(1-auc_b, auc_b)
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
            
            if "str" in str(type(self.X[0,feature])):
                index_left = indexs[self.X[indexs, feature] == split_value]
                index_right = indexs[self.X[indexs, feature] != split_value]
            else:
                index_left = indexs[self.X[indexs, feature] < split_value]
                index_right = indexs[self.X[indexs, feature] >= split_value]
                
            if (len(index_left)==0) or (len(index_right)==0):
                if self.criterion == "auc":
                    score = 0
                else:
                    score = -np.inf
            elif self.criterion == "auc":
                auc_y = get_auc_y(index_left, indexs)
                auc_b = get_auc_b(index_left, indexs)
                score = (1-self.orthogo_coef)*auc_y - self.orthogo_coef*auc_b
            
            elif self.criterion == "faht":
                
                n = len(indexs)
                pos_n = sum(self.y[indexs]==self.y_pos)
                neg_n = n - pos_n
                pos_prob = pos_n/n
                neg_prob = neg_n/n
                entropy_parent = entropy([pos_prob, neg_prob], base=2)

                n_left = len(index_left)
                pos_n_left = sum(self.y[index_left]==self.y_pos)
                neg_n_left = n_left - pos_n_left
                pos_prob_left = pos_n_left/n_left
                neg_prob_left = neg_n_left/n_left
                entropy_left = entropy([pos_prob_left, neg_prob_left], base=2)

                n_right = len(index_right)
                pos_n_right = sum(self.y[index_right]==self.y_pos)
                neg_n_right = n_right - pos_n_right
                pos_prob_right = pos_n_right/n_right
                neg_prob_right = neg_n_right/n_right
                entropy_right = entropy([pos_prob_right, neg_prob_right], base=2)

                ig = entropy_parent - (
                    (n_left/n) * entropy_left + (n_right/n) * entropy_right
                )
                
                dr = sum((self.b[indexs]==self.b_neg) & (self.y[indexs]==self.y_neg)) # deprived rejected
                dg = sum((self.b[indexs]==self.b_neg) & (self.y[indexs]==self.y_pos)) # deprived granted
                fr = sum((self.b[indexs]==self.b_pos) & (self.y[indexs]==self.y_neg)) # favoured rejected
                fg = sum((self.b[indexs]==self.b_pos) & (self.y[indexs]==self.y_pos)) # favoured granted
                disc = (fg/(fg+fr)) - (dg/(dg+dr))

                dr_left = sum((self.b[index_left]==self.b_neg) & (self.y[index_left]==self.y_neg)) # deprived rejected
                dg_left = sum((self.b[index_left]==self.b_neg) & (self.y[index_left]==self.y_pos)) # deprived granted
                fr_left = sum((self.b[index_left]==self.b_pos) & (self.y[index_left]==self.y_neg)) # favoured rejected
                fg_left = sum((self.b[index_left]==self.b_pos) & (self.y[index_left]==self.y_pos)) # favoured granted                
                if (fg_left+fr_left)==0:
                    disc_left = (dg_left/(dg_left+dr_left))
                elif (dg_left+dr_left)==0:
                    disc_left = (fg_left/(fg_left+fr_left)) 
                else:
                    disc_left = (fg_left/(fg_left+fr_left)) - (dg_left/(dg_left+dr_left))
                
                dr_right = sum((self.b[index_right]==self.b_neg) & (self.y[index_right]==self.y_neg)) # deprived rejected
                dg_right = sum((self.b[index_right]==self.b_neg) & (self.y[index_right]==self.y_pos)) # deprived granted
                fr_right = sum((self.b[index_right]==self.b_pos) & (self.y[index_right]==self.y_neg)) # favoured rejected
                fg_right = sum((self.b[index_right]==self.b_pos) & (self.y[index_right]==self.y_pos)) # favoured granted
                if (fg_right+fr_right)==0:
                    disc_right = (dg_right/(dg_right+dr_right))
                elif (dg_right+dr_right)==0:
                    disc_right = (fg_right/(fg_right+fr_right)) 
                else:
                    disc_right = (fg_right/(fg_right+fr_right)) - (dg_right/(dg_right+dr_right))
                
                fg = abs(disc) - ( (n_left/n) * abs(disc_left) + (n_right/n) * abs(disc_right))
                if (fg==0):
                    fg = 1 # FIG=IG*FG, and when FG=0, authors state FIG=IG --> FG=1 since FIG=IG*FG -> FIG=IG

                elif np.isnan(fg):
                    fg = -np.inf
                    ig = 1
                    
                score = ig * fg # fair information gain
                
            elif self.criterion in ["entropy", "ig"]:
                
                n = len(indexs)
                pos_n = sum(self.y[indexs]==self.y_pos)
                neg_n = n - pos_n
                pos_prob = pos_n/n
                neg_prob = neg_n/n
                entropy_parent = entropy([pos_prob, neg_prob], base=2)

                n_left = len(index_left)
                pos_n_left = sum(self.y[index_left]==self.y_pos)
                neg_n_left = n_left - pos_n_left
                pos_prob_left = pos_n_left/n_left
                neg_prob_left = neg_n_left/n_left
                entropy_left = entropy([pos_prob_left, neg_prob_left], base=2)

                n_right = len(index_right)
                pos_n_right = sum(self.y[index_right]==self.y_pos)
                neg_n_right = n_right - pos_n_right
                pos_prob_right = pos_n_right/n_right
                neg_prob_right = neg_n_right/n_right
                entropy_right = entropy([pos_prob_right, neg_prob_right], base=2)

                ig = entropy_parent - (
                    (n_left/n) * entropy_left + (n_right/n) * entropy_right
                )
                score = ig # information gain
            
            elif self.criterion == "fg":
                
                n = len(indexs)
                dr = sum((self.b[indexs]==self.b_neg) & (self.y[indexs]==self.y_neg)) # deprived rejected
                dg = sum((self.b[indexs]==self.b_neg) & (self.y[indexs]==self.y_pos)) # deprived granted
                fr = sum((self.b[indexs]==self.b_pos) & (self.y[indexs]==self.y_neg)) # favoured rejected
                fg = sum((self.b[indexs]==self.b_pos) & (self.y[indexs]==self.y_pos)) # favoured granted
                disc = (fg/(fg+fr)) - (dg/(dg+dr))
                
                n_left = len(index_left)
                dr_left = sum((self.b[index_left]==self.b_neg) & (self.y[index_left]==self.y_neg)) # deprived rejected
                dg_left = sum((self.b[index_left]==self.b_neg) & (self.y[index_left]==self.y_pos)) # deprived granted
                fr_left = sum((self.b[index_left]==self.b_pos) & (self.y[index_left]==self.y_neg)) # favoured rejected
                fg_left = sum((self.b[index_left]==self.b_pos) & (self.y[index_left]==self.y_pos)) # favoured granted                
                if (fg_left+fr_left)==0:
                    disc_left = (dg_left/(dg_left+dr_left))
                elif (dg_left+dr_left)==0:
                    disc_left = (fg_left/(fg_left+fr_left)) 
                else:
                    disc_left = (fg_left/(fg_left+fr_left)) - (dg_left/(dg_left+dr_left))
                
                n_right = len(index_right)
                dr_right = sum((self.b[index_right]==self.b_neg) & (self.y[index_right]==self.y_neg)) # deprived rejected
                dg_right = sum((self.b[index_right]==self.b_neg) & (self.y[index_right]==self.y_pos)) # deprived granted
                fr_right = sum((self.b[index_right]==self.b_pos) & (self.y[index_right]==self.y_neg)) # favoured rejected
                fg_right = sum((self.b[index_right]==self.b_pos) & (self.y[index_right]==self.y_pos)) # favoured granted
                if (fg_right+fr_right)==0:
                    disc_right = (dg_right/(dg_right+dr_right))
                elif (dg_right+dr_right)==0:
                    disc_right = (fg_right/(fg_right+fr_right)) 
                else:
                    disc_right = (fg_right/(fg_right+fr_right)) - (dg_right/(dg_right+dr_right))
                
                fg = abs(disc) - (
                    (n_left/n)*abs(disc_left) + (n_right/n)*abs(disc_right)
                )
#                 print(fg)
#                 if (fg==0):
#                     fg = 1 # FIG=IG*FG, and when FG=0, authors state FIG=IG --> FG=1 since FIG=IG*FG -> FIG=IG
#                 if np.isnan(fg):
#                     fg = 0
                if np.isnan(fg):
                    fg = -np.inf
                score = fg # fairness gain
            
            elif "kamiran" in self.criterion:
                n = len(indexs)
                pos_n = sum(self.y[indexs]==self.y_pos)
                neg_n = n - pos_n
                pos_prob = pos_n/n
                neg_prob = neg_n/n
                entropy_parent = entropy([pos_prob, neg_prob], base=2)

                n_left = len(index_left)
                pos_n_left = sum(self.y[index_left]==self.y_pos)
                neg_n_left = n_left - pos_n_left
                pos_prob_left = pos_n_left/n_left
                neg_prob_left = neg_n_left/n_left
                entropy_left = entropy([pos_prob_left, neg_prob_left], base=2)

                n_right = len(index_right)
                pos_n_right = sum(self.y[index_right]==self.y_pos)
                neg_n_right = n_right - pos_n_right
                pos_prob_right = pos_n_right/n_right
                neg_prob_right = neg_n_right/n_right
                entropy_right = entropy([pos_prob_right, neg_prob_right], base=2)

                igc = entropy_parent - (
                    (n_left/n) * entropy_left + (n_right/n) * entropy_right
                )

                pos_n = sum(self.b[indexs]==self.b_pos)
                neg_n = n - pos_n
                pos_prob = pos_n/n
                neg_prob = neg_n/n
                entropy_parent = entropy([pos_prob, neg_prob], base=2)

                n_left = len(index_left)
                pos_n_left = sum(self.b[index_left]==self.b_pos)
                neg_n_left = n_left - pos_n_left
                pos_prob_left = pos_n_left/n_left
                neg_prob_left = neg_n_left/n_left
                entropy_left = entropy([pos_prob_left, neg_prob_left], base=2)

                n_right = len(index_right)
                pos_n_right = sum(self.b[index_right]==self.b_pos)
                neg_n_right = n_right - pos_n_right
                pos_prob_right = pos_n_right/n_right
                neg_prob_right = neg_n_right/n_right
                entropy_right = entropy([pos_prob_right, neg_prob_right], base=2)
                
                igs = entropy_parent - (
                    (n_left/n) * entropy_left + (n_right/n) * entropy_right
                )
                
                if "add" in self.criterion:
                    score = igc + igs
                
                if "sub" in self.criterion:
                    score = igc - igs
                    
                if "div" in self.criterion:
                    score = igc / igs
                    
            return score    
                
        # return best (sscore, feature, split_value) dependant on criterion and indexs
        def get_best_split(indexs):
            if self.criterion=="auc":
                best_score = 0
            else:
                best_score = -np.inf
            # only positive scores are desirable
            # if negative score, then b_auc > s_auc which we don't want
            candidate_splits = get_candidate_splits(indexs)
            for feature in candidate_splits:
                for split_value in candidate_splits[feature]:
                    score = evaluate_split(feature, split_value, indexs)
                    #print(score)
                    if score > best_score:
                        best_score = score
                        best_feature = feature
                        best_split_value = split_value
            if (best_score==-np.inf):
                best_feature, best_split_value = np.nan, np.nan
            if (self.criterion=="auc") and (best_score==0):
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
                #print(new_score)
                if new_score==-np.inf: ## in case no more feature values exist for splitting
                    return indexs
                
                if (self.criterion=="auc") and (new_score<=0):
                    return indexs
#                 if new_score <= old_score:
#                     return indexs
                
                ##print(indexs)
                left_indexs = indexs[self.X[indexs, feature]<split_value]
                right_indexs = indexs[self.X[indexs, feature]>=split_value]
                
                if (len(left_indexs)==0) or (len(right_indexs)==0):
                    return indexs
                
                else:
                    tree[(feature, split_value)] = {
                        "<": build_tree(left_indexs, step=copy(step), old_score=copy(old_score), new_score=copy(new_score)),
                        ">=":  build_tree(right_indexs, step=copy(step), old_score=copy(old_score), new_score=copy(new_score))
                    }

                    return tree
        
        self.tree = build_tree(self.samples)
        del self.X
        self.is_fit=True   
       
    def predict_proba(self, X):

        def get_probas_dict(tree, X, indexs=np.array([]), probas_dict={}):

            indexs = np.array(range(X.shape[0])) if len(indexs)==0 else indexs
            if type(tree)==type({}):
                feature, value = list(tree.keys())[0]
                left_indexs = indexs[X[indexs, feature]<value]
                sub_tree = tree[(feature, value)]["<"]
                probas_dict = get_probas_dict(sub_tree, X, left_indexs, probas_dict)
                right_indexs = indexs[X[indexs, feature]>=value]
                sub_tree = tree[(feature, value)][">="]
                probas_dict = get_probas_dict(sub_tree, X, right_indexs, probas_dict)
                return probas_dict

            else:
                index = copy(tree)
                sub_y = self.y[index]
                proba = sum(sub_y)/len(sub_y)
                if proba in probas_dict:
                    probas_dict[proba] += indexs.tolist()
                else:
                    probas_dict[proba] = indexs.tolist()
                return probas_dict

        proba = np.repeat(0.0, X.shape[0])
        probas_dict = get_probas_dict(self.tree, X)
        for proba_value in probas_dict:
            proba_index = np.array(probas_dict[proba_value])
            proba[proba_index] =  proba_value

        return proba
    
    def predict(self, X):
        
        def predict_proba(X):

            def get_probas_dict(tree, X, indexs=np.array([]), probas_dict={}):

                indexs = np.array(range(X.shape[0])) if len(indexs)==0 else indexs
                if type(tree)==type({}):
                    feature, value = list(tree.keys())[0]
                    left_indexs = indexs[X[indexs, feature]<value]
                    sub_tree = tree[(feature, value)]["<"]
                    probas_dict = get_probas_dict(sub_tree, X, left_indexs, probas_dict)
                    right_indexs = indexs[X[indexs, feature]>=value]
                    sub_tree = tree[(feature, value)][">="]
                    probas_dict = get_probas_dict(sub_tree, X, right_indexs, probas_dict)
                    return probas_dict

                else:
                    index = copy(tree)
                    sub_y = self.y[index]
                    proba = sum(sub_y)/len(sub_y)
                    probas_dict[indexs] = proba
                    return probas_dict

            proba = np.repeat(0.0, X.shape[0])
            probas_dict = get_probas_dict(self.tree, X)
            for indexs in probas_dict:
                proba_value = probas_dict[indexs]
                proba[indexs] =  proba_value

            return proba
        
        probas = predict_proba(X)
        predicts = np.repeat(0, X.shape[0])
        predicts[probas>=self.pred_th] = 1
        
        return predicts
    
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
    def __init__(self, n_estimators=501, n_jobs=-1,
        n_bins=100, min_leaf=3, max_depth=5, n_samples=1.0, n_features="auto", boot_replace=True, random_state=42,
        criterion="auc", bias_method="avg", compound_bias_method="avg", orthogo_coef=.6
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
                   -> str:
                       -> "auto"/"sqrt": sqrt of features is used
                       -> "log"/"log2": log2 of features is used
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
        
        if self.criterion not in ["entropy", "auc", "faht", "ig", "fg", "kamiran_add", "kamiran_div", "kamiran_sub"]:
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
#         if ("int" not in str(type(self.n_features))) and ("float" not in str(type(self.n_features))):
#             raise Exception("n_features must be an int or float, not " + str(type(self.n_features)))
        if ("int" not in str(type(self.orthogo_coef))) and ("float" not in str(type(self.orthogo_coef))):
            raise Exception("orthogo_coef must be an int or float, not " + str(type(self.orthogo_coef)))
        
        
        # Generating BCDForest
        dts = [
            BiasConstraintDecisionTreeClassifier(
                n_bins=self.n_bins,
                min_leaf=self.min_leaf,
                max_depth=self.max_depth,
                n_samples=self.n_samples,
                criterion=self.criterion,
                random_state=self.random_state+i,
                n_features=self.n_features,
                bias_method=self.bias_method,
                orthogo_coef=self.orthogo_coef,
                boot_replace=self.boot_replace,
                compound_bias_method=self.compound_bias_method,
            )
            for i in range(self.n_estimators)
        ]
        self.trees = dts
        
    def fit(self, X, y, s):
      
        def batch(iterable, n_jobs=1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count() # - 1 # -1 so that our laptop doesn't freeze
            l = len(iterable)
            n = int(np.ceil(l / n_jobs))
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        def fit_trees_parallel(i, dt_batches, X, y, s):
            dt_batch = dt_batches[i]
            fit_dt_batch = []
            for dt in tqdm(dt_batch, desc=str(i)):
                dt.fit(X, y, s)
                fit_dt_batch.append(dt)
            return fit_dt_batch
    
        dts = self.trees
        dt_batches = list(batch(dts, n_jobs=self.n_jobs))
        fit_dt_batches = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_trees_parallel)(
                i, #copy(i),
                dt_batches, #copy(dt_batches),
                X, #copy(X),
                y, #copy(y),
                s, #copy(s)
            ) for i in (range(len(copy(dt_batches))))
        )
        fit_dts = [tree for fit_dt_batch in fit_dt_batches for tree in fit_dt_batch]
        self.trees = fit_dts
        self.fit = True
    
    def predict_proba(self, X):
        def predict_proba_parallel(dt_batch, X, i):
            probas = []
            for tree in dt_batch:
                probas.append(tree.predict_proba(X))
            return np.array(probas)
        
        def batch(iterable, n_jobs=1):
            if n_jobs==-1:
                n_jobs = multiprocessing.cpu_count() # - 1 # -1 so that our laptop doesn't freeze
            l = len(iterable)
            n = ceil(l / n_jobs)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        
        if not self.fit:
            warnings.warn("Forest has not been fit(X,y,s)")
        
        
        else:
            dt_batches = list(batch(self.trees, n_jobs=self.n_jobs))
            
            y_preds = Parallel(n_jobs=self.n_jobs)(
                delayed(predict_proba_parallel)(
                    copy(dt_batches[i]),
                    copy(X),
                    copy(i),
                ) for i in range(len(dt_batches))
            )
            
            y_prob = y_preds[0]
            for i in range(1, len(y_preds)):
                y_prob = np.concatenate(
                    (y_prob, y_preds[i]),
                    axis=0
                )
            return np.mean(y_prob, axis=0)
    
    def predict(self, X):
        def predict_parallel(tree, X):
            return tree.predict(X)
        if not self.fit:
            warnings.warn("Forest has not been fit(X,y,s)")

        else:
            # Predicting
            y_preds = Parallel(n_jobs=self.n_jobs)(
                delayed(predict_parallel)(
                    tree, X
                ) for tree in self.trees
            )
            y_preds = np.array(y_preds)
            # adding an "extra tree" with all positives so that ties are considered positive (just like >= th)
            predictions = mode(
                np.concatenate((
                    y_preds,
                    np.repeat(self.y_pos, y_preds.shape[1]).reshape(1, y_preds.shape[1])
                ), axis=0))[0][0]
            return predictions
    
def run_regression(
    X="X", y="y", s="s", Scaler=RS,
    cov_coefs=sorted(set(np.linspace(0,1,11).tolist() + np.geomspace(1e-2,1,11).tolist())), random_state=42,
    Splitter=StratifiedGroupKFold, n_splits=2, n_bins=1, degree=1, perf_measure=mean_squared_error, plot=True,return_output=False
):
    """
    if return_output: cov_test_measures, cov_train_measures, cov_test_covariances, cov_train_covariances
    """
    cov_test_measures = []
    cov_train_measures = []
    cov_test_covariances = []
    cov_train_covariances = []
    for cov_coef in tqdm_n(cov_coefs):
        test_measures = []
        train_measures = []
        test_covariances = []
        train_covariances = []
        if "Stratified" in str(Splitter):
            splitter = Splitter(n_splits=n_splits, random_state=random_state)
        else:
            splitter = Splitter(n_splits=n_splits)
        y_split = pd.qcut(y, n_bins, labels=np.array(range(n_bins)))
        for train_idx, test_idx in splitter.split(X,y_split,s):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            s_train, s_test = s[train_idx], s[test_idx]

#             scaler_y = Scaler().fit(y_train.reshape(y_train.shape[0],1))
#             y_test = scaler_y.transform(y_test.reshape(y_test.shape[0],1)).ravel()
#             y_train = scaler_y.transform(y_train.reshape(y_train.shape[0],1)).ravel()

            z_test = pd.get_dummies(s_test) - pd.get_dummies(s_test).mean()
            z_test[::] = 1 if z_test.shape[1]==1 else z_test.values
            z_train = pd.get_dummies(s_train) - pd.get_dummies(s_train).mean()

            pf = PF(degree=degree, include_bias=False).fit(X_train)
            X_test = pf.transform(X_test)
            X_train = pf.transform(X_train)

            scaler_X = Scaler().fit(X_train)
            X_test = scaler_X.transform(X_test)
            X_train = scaler_X.transform(X_train)

            reg = CovarianceConstraintLinearRegression(
                cov_coefficient=cov_coef
            )
            reg.fit(X_train, y_train, s_train)

            train_covariance = []
            for s_uni in np.unique(s_train):
                uni_train_covariance = abs(np.mean(reg.predict(X_train) * z_train[s_uni]))
                train_covariance.append(uni_train_covariance)
            train_covariance = np.mean(train_covariance)
            train_covariances.append(train_covariance)

            test_covariance = []
            for s_uni in np.unique(s_test):
                uni_test_covariance = abs(np.mean(reg.predict(X_test) * z_test[s_uni]))
                test_covariance.append(uni_test_covariance)
            test_covariance = np.mean(test_covariance)
            test_covariances.append(test_covariance)

            test_measure = perf_measure(y_test, reg.predict(X_test))
            train_measure = perf_measure(y_train, reg.predict(X_train))
            test_measures.append(test_measure)
            train_measures.append(train_measure)
        cov_test_measures.append(test_measures)
        cov_train_measures.append(train_measures)
        cov_test_covariances.append(test_covariances)
        cov_train_covariances.append(train_covariances)

    if plot:
        fig, axs = plt.subplots(1,2, figsize=(11,3), dpi=150, sharex=True)
        ax = axs[0]
        sb.lineplot(x=cov_coefs, y=np.mean(cov_test_measures, axis=1), label="test", ax=ax, color="C0")
        ax.fill_between(
            x=cov_coefs, color="C0", alpha=0.1,
            y1=np.mean(cov_test_measures, axis=1)+np.std(cov_test_measures, axis=1),
            y2=np.mean(cov_test_measures, axis=1)-np.std(cov_test_measures, axis=1),
        )

        sb.lineplot(x=cov_coefs, y=np.mean(cov_train_measures, axis=1), label="train", ax=ax, color="C1")
        ax.fill_between(
            x=cov_coefs, color="C1", alpha=0.1,
            y1=np.mean(cov_train_measures, axis=1)+np.std(cov_train_measures, axis=1),
            y2=np.mean(cov_train_measures, axis=1)-np.std(cov_train_measures, axis=1),
        )
        ax.set_xlabel("Covariance coefficient\n(relative allowed covariance)")
        ax.set_ylabel("Performance (" +str(perf_measure).split(" ")[1]+")")
        ax.grid()

        ax2 = axs[1]
        sb.lineplot(x=cov_coefs, y=np.mean(cov_test_covariances, axis=1), label="test", ax=ax2, color="C2")
        ax2.fill_between(
            x=cov_coefs, color="C2", alpha=0.1,
            y1=np.mean(cov_test_covariances, axis=1)+np.std(cov_test_covariances, axis=1),
            y2=np.mean(cov_test_covariances, axis=1)-np.std(cov_test_covariances, axis=1),
        )

        sb.lineplot(x=cov_coefs, y=np.mean(cov_train_covariances, axis=1), label="train", ax=ax2, color="C4")
        ax2.fill_between(
            x=cov_coefs, color="C4", alpha=0.1,
            y1=np.mean(cov_train_covariances, axis=1)+np.std(cov_train_covariances, axis=1),
            y2=np.mean(cov_train_covariances, axis=1)-np.std(cov_train_covariances, axis=1),
        )

        ax2.set_xlabel("Covariance coefficient\n(relative allowed covariance)")
        ax2.set_ylabel("Covariance")
        ax2.grid()
        plt.suptitle("Regression", x=.5125)
        plt.show()
        
    if return_output == True:
        return cov_test_measures, cov_train_measures, cov_test_covariances, cov_train_covariances

def run_classification(
    X="X", y="y", s="s", Scaler=SS, sample_weights=None,
    cov_coefs=sorted(set(np.linspace(0,1,21).tolist())), random_state=42,
    Splitter=StratifiedGroupKFold, n_splits=2, degree=1, perf_measure=roc_auc_score, plot=True,return_output=False
):
    """
    if return_output: cov_test_measures, cov_train_measures, cov_test_covariances, cov_train_covariances
    """
    cov_test_measures = []
    cov_train_measures = []
    cov_test_covariances = []
    cov_train_covariances = []
    for cov_coef in tqdm_n(cov_coefs):
        test_measures = []
        train_measures = []
        test_covariances = []
        train_covariances = []
        if "Stratified" in str(Splitter):
            splitter = Splitter(n_splits=n_splits, random_state=random_state)
        else:
            splitter = Splitter(n_splits=n_splits)
        for train_idx, test_idx in splitter.split(X,y,s):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            s_train, s_test = s[train_idx], s[test_idx]

            z_test = pd.get_dummies(s_test) - pd.get_dummies(s_test).mean()
            z_test[::] = 1 if z_test.shape[1]==1 else z_test.values
            z_train = pd.get_dummies(s_train) - pd.get_dummies(s_train).mean()
            
            pf = PF(degree=degree, include_bias=False).fit(X_train)
            X_test = pf.transform(X_test)
            X_train = pf.transform(X_train)
            
            scaler_X = Scaler().fit(X_train)
            X_test = scaler_X.transform(X_test)
            X_train = scaler_X.transform(X_train)
            
            if type(sample_weights)==type(np.array([])):
                weights=sample_weights
                                          
            elif sample_weights=="kliep":
                kliep = DensityRatioEstimator()
                kliep.fit(X_train, X_test) # keyword arguments are X_train and X_test
                weights = kliep.predict(X_train)
            
            else:
                weights=None
            
            counter = 0
            success = 0
            while not success:
                try:
                    clf = CovarianceConstraintLogisticRegression(
                        cov_coefficient=cov_coef
                    )
                    clf.fit(X_train, y_train, s_train, weights=weights)
                    success = 1
                except:
                    print("covariance-coefficient " + cov_coef + " failed, rounding further")
                    cov_coef = round(cov_coef, 5-counter)
                    counter += 1
                    
            train_covariance = []
            for s_uni in np.unique(s_train):
                uni_train_covariance = abs(np.mean(clf.predict(X_train) * z_train[s_uni]))
                train_covariance.append(uni_train_covariance)
            train_covariance = np.mean(train_covariance)
            train_covariances.append(train_covariance)

            test_covariance = []
            for s_uni in np.unique(s_test):
                uni_test_covariance = abs(np.mean(clf.predict(X_test) * z_test[s_uni]))
                test_covariance.append(uni_test_covariance)
            test_covariance = np.mean(test_covariance)
            test_covariances.append(test_covariance)

            test_measure = perf_measure(y_test, clf.predict(X_test))
            train_measure = perf_measure(y_train, clf.predict(X_train))
            test_measures.append(test_measure)
            train_measures.append(train_measure)
        cov_test_measures.append(test_measures)
        cov_train_measures.append(train_measures)
        cov_test_covariances.append(test_covariances)
        cov_train_covariances.append(train_covariances)

    if plot:
        fig, axs = plt.subplots(1,2, figsize=(11,3), dpi=150, sharex=True)
        ax = axs[0]
        sb.lineplot(x=cov_coefs, y=np.mean(cov_test_measures, axis=1), label="test", ax=ax, color="C0")
        ax.fill_between(
            x=cov_coefs, color="C0", alpha=0.1,
            y1=np.mean(cov_test_measures, axis=1)+np.std(cov_test_measures, axis=1),
            y2=np.mean(cov_test_measures, axis=1)-np.std(cov_test_measures, axis=1),
        )

        sb.lineplot(x=cov_coefs, y=np.mean(cov_train_measures, axis=1), label="train", ax=ax, color="C1")
        ax.fill_between(
            x=cov_coefs, color="C1", alpha=0.1,
            y1=np.mean(cov_train_measures, axis=1)+np.std(cov_train_measures, axis=1),
            y2=np.mean(cov_train_measures, axis=1)-np.std(cov_train_measures, axis=1),
        )
        ax.set_xlabel("Covariance coefficient\n(relative allowed covariance)")
        ax.set_ylabel("Performance (" +str(perf_measure).split(" ")[1]+")")
        ax.grid()

        ax2 = axs[1]
        sb.lineplot(x=cov_coefs, y=np.mean(cov_test_covariances, axis=1), label="test", ax=ax2, color="C2")
        ax2.fill_between(
            x=cov_coefs, color="C2", alpha=0.1,
            y1=np.mean(cov_test_covariances, axis=1)+np.std(cov_test_covariances, axis=1),
            y2=np.mean(cov_test_covariances, axis=1)-np.std(cov_test_covariances, axis=1),
        )

        sb.lineplot(x=cov_coefs, y=np.mean(cov_train_covariances, axis=1), label="train", ax=ax2, color="C4")
        ax2.fill_between(
            x=cov_coefs, color="C4", alpha=0.1,
            y1=np.mean(cov_train_covariances, axis=1)+np.std(cov_train_covariances, axis=1),
            y2=np.mean(cov_train_covariances, axis=1)-np.std(cov_train_covariances, axis=1),
        )

        ax2.set_xlabel("Covariance coefficient\n(relative allowed covariance)")
        ax2.set_ylabel("Covariance")
        ax2.grid()
        plt.suptitle("Classification", x=.5125)
        plt.show()
    
    if return_output == True:
        return cov_test_measures, cov_train_measures, cov_test_covariances, cov_train_covariances

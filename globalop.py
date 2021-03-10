import numpy as np
import multiprocessing
from pprint import pprint
from copy import deepcopy as copy
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score

class DecisionTreeClassifier():

    def __init__(self,
        criterion="auc_div", n_bins=2, max_depth=2, bootstrap=False, max_features="sqrt", orthogonality=0.5, score_weight=False, random_state=42
    ):
        self.n_bins = n_bins
        self.criterion = criterion
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.score_weight = score_weight
        self.random_state = random_state
        self.orthogonality = orthogonality
        
    def fit(self, X, y, s):
        
        np.random.seed(self.random_state)
        features = np.array(range(X.shape[1]))
        indexs = np.array(range(len(y)))
        if self.bootstrap:
            indexs = np.random.choice(indexs, size=len(indexs), replace=True)
        tree = {
            "id": "base",
            "indexs": indexs,
            "probas": sum(y[indexs]==1)/len(indexs)
        }
        
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

        def get_probas(tree, output=np.array([])):
            tree = copy(tree)
            if "split" in tree:
                node_0 = tree[0]
                node_1 = tree[1]
                output_0 = get_probas(node_0, output)
                output_1 = get_probas(node_1, output)
                output = np.concatenate((output_0, output_1))
                return copy(output)
            else:
                return copy(np.repeat(tree["probas"], len(tree["indexs"])))

        def get_indexs(tree, output=np.array([])):
            tree=copy(tree)
            if "split" in tree:
                node_0 = tree[0]
                node_1 = tree[1]
                output_0 = copy(get_indexs(node_0, output))
                output_1 = copy(get_indexs(node_1, output))
                output = np.concatenate((output_0, output_1))
                return copy(output)
            else:
                return copy(tree["indexs"])
        
        def get_weights(tree, output=np.array([])):
            tree = copy(tree)
            if "split" in tree:
                node_0 = tree[0]
                node_1 = tree[1]
                output_0 = get_weights(node_0, output)
                output_1 = get_weights(node_1, output)
                output = np.concatenate((output_0, output_1))
                return copy(output)
            else:
                return copy(np.repeat(len(tree["indexs"]), len(tree["indexs"])))
        
        def get_score(tree):
            # self.y
            tree = copy(tree)
            indexs = copy(get_indexs(tree))
            probas = copy(get_probas(tree))
            y_true = y[indexs]
            s_true = s[indexs]
            if self.score_weight:
                score_weight = get_weights(tree)
                if len(np.unique(score_weight)) == 1:
                    score_weight = None
                y_auc = roc_auc_score(y_true, probas, sample_weight=score_weight)
                s_auc = roc_auc_score(s_true, probas, sample_weight=score_weight)
            else:
                y_auc = roc_auc_score(y_true, probas)
                s_auc = roc_auc_score(s_true, probas)
            s_auc = max(1-s_auc, s_auc)
            if self.criterion=="auc_sub":
                score = (1-self.orthogonality)*y_auc - self.orthogonality*s_auc
            elif self.criterion=="auc_div":
                score = y_auc / s_auc
            return copy(score)

        def get_candidate_splits(indexs):
            # self.features
            # self.X
            indexs = copy(indexs)
            candidate_splits = {}
            chosen_features = choose_features()
            #print(chosen_features)
            for feature in chosen_features:
                candidate_splits[feature] = {}
                values = np.unique(X[indexs, feature])
#                 ###################################################
#                 # making values into bin_split_points
#                 n_unique = len(values)
#                 if (n_unique-1) >= self.n_bins:
#                     lo = 1/self.n_bins
#                     hi = lo * (self.n_bins-1)
#                     quantiles = np.linspace(lo, hi, self.n_bins-1)
#                     values = list(np.quantile(values, q=quantiles))
                ###################################################    
                n_unique = len(values)
                if (n_unique) > self.n_bins:
                    lo = 1/self.n_bins
                    hi = lo * (self.n_bins-1)
                    quantiles = np.linspace(lo, hi, self.n_bins-1)
                    values = list(np.quantile(values, q=quantiles))
                
                
                for value in values:
                    indexs_0 = indexs[X[indexs, feature] < value]
                    indexs_1 = indexs[X[indexs, feature] >= value]
                    if (len(indexs_0) > 0) and (len(indexs_1) > 0):
                        node_0 = {
                            "indexs": indexs_0,
                            "probas": sum(y[indexs_0]==1)/len(indexs_0)
                        }

                        node_1 = {
                            "indexs": indexs_1,
                            "probas": sum(y[indexs_1]==1)/len(indexs_1)
                        }
                        candidate_splits[feature][value] = {0: node_0, 1: node_1}
                if candidate_splits[feature] == {}:
                    # if all Xs have the same feat value
                    del candidate_splits[feature]
            return copy(candidate_splits)

        def get_leaf_ids(tree, ids=[]):
            # apparently, when working with tuples
            # if you don't deepcopy, you can't recurse
            # that's 1.5 hour i'm not getting back...
            ids = copy(ids)
            tree = copy(tree)
            if "split" in tree:
                node_0 = tree[0]
                node_1 = tree[1]
                del tree
                id_0 = get_leaf_ids(node_0, copy(ids))
                del node_0
                id_1 = get_leaf_ids(node_1, copy(ids))
                del node_1
                ids += id_0
                del id_0
                ids += id_1
                del id_1
                return ids
            else:
                return [tree["id"]]

        def get_leaf(tree, leaf_id):
            tree = copy(tree)
            leaf = copy(tree)
            leaf_id = copy(leaf_id)
            if leaf_id != "base":
                for i in range(len(leaf_id)):
                    path = leaf_id[i]
                    leaf = leaf[path]
            return copy(leaf)

        def get_leaf_id_candidate_splits(tree):
            tree = copy(tree)
            leaf_id_candidate_splits = {}
            best_score = 0
            leaf_ids = get_leaf_ids(tree)
            #print(leaf_ids)
            for leaf_id in leaf_ids:
                #print(leaf_id)
                leaf = copy(get_leaf(tree, leaf_id))
                indexs = leaf["indexs"]
                candidate_splits = get_candidate_splits(indexs)
                leaf_id_candidate_splits[leaf_id] = candidate_splits
            #print(leaf_id_candidate_splits)
            return copy(leaf_id_candidate_splits)

        def get_new_tree_with_add_best(tree, best):
            # returns a tree with the added best split
            tree = copy(tree)
            best = copy(best)

            leaf_id = copy(best[0])
            split = copy(best[1]) # dictionary just like a node with 2 leaves
            feature = best[2]
            value = best[3]

            #print(leaf_id)
            leaf = tree
            if leaf_id != "base":
                for i in range(len(leaf_id)):
                    #print("in loop")
                    path = leaf_id[i]
                    leaf = leaf[path]

            #print(leaf["id"])
            #pprint(leaf)

            leaf["split"] = copy((feature, value))
            leaf[0] = copy(split[0])
            leaf[1] = copy(split[1])
            if leaf_id=="base":
                leaf[0]["id"] = 0,
                leaf[1]["id"] = 1,
                #leaf[0]["from"] = copy((feature, value))
                #leaf[1]["from"] = copy((feature, value))
            else:
                leaf[0]["id"] = copy(tuple([path for path in leaf_id]+[0]))
                leaf[1]["id"] = copy(tuple([path for path in leaf_id]+[1]))
                #leaf[0]["from"] = copy((feature, value))
                #leaf[1]["from"] = copy((feature, value))
            
            # we won't delete this info when
            # it might be useful to do pruning
            # since we need to know the probs
            del leaf["indexs"]
            del leaf["probas"]
            return copy(tree)

        def get_best_leaf_id_split_feature_value_score(tree, leaf_id_candidate_splits):
            tree = copy(tree)
            leaf_id_candidate_splits = copy(leaf_id_candidate_splits)
            base_score = copy(get_score(tree))
            best_score = copy(get_score(tree))
            for leaf_id in leaf_id_candidate_splits:
                for feature in leaf_id_candidate_splits[leaf_id]:
                    for value in leaf_id_candidate_splits[leaf_id][feature]:
                        split = leaf_id_candidate_splits[leaf_id][feature][value]
                        #print(split)
                        best = leaf_id, split, feature, value
                        new_tree = get_new_tree_with_add_best(copy(tree), copy(best))
                        new_score = get_score(new_tree)
                        #print(best, new_score)
                        #print(new_score)
                        if (
                            (new_score > best_score) and (len(leaf_id)<self.max_depth)
                        ) or (
                            (new_score > best_score) and (leaf_id=="base")
                        ):
                            best_score = new_score
                            best_leaf_id = leaf_id
                            best_split = split
                            best_feature = feature
                            best_value = value

            if base_score == best_score:
                return copy((0, 0, 0, 0, -np.inf))
            else:
                return copy((best_leaf_id, best_split, best_feature, best_value, best_score))

        def update_leaf_id_candidate_splits(leaf_id_candidate_splits, best):
            leaf_id_candidate_splits = copy(leaf_id_candidate_splits)
            best = copy(best)
            leaf_id = best[0]
            split = best[1] # dictionary just like a node with 2 leaves
            feature = best[2]
            value = best[3]
            # removing old_leaf
            del leaf_id_candidate_splits[leaf_id]
            if leaf_id != "base":
                new_leaf_id = copy(tuple([path for path in leaf_id]+[0]))
                indexs = split[0]["indexs"]
                candidate_splits = get_candidate_splits(indexs)
                leaf_id_candidate_splits[new_leaf_id] = candidate_splits

                new_leaf_id = copy(tuple([path for path in leaf_id]+[1]))
                indexs = split[1]["indexs"]
                candidate_splits = get_candidate_splits(indexs)
                leaf_id_candidate_splits[new_leaf_id] = candidate_splits

            else:
                new_leaf_id = copy(tuple([0]))
                indexs = split[0]["indexs"]
                candidate_splits = get_candidate_splits(indexs)
                leaf_id_candidate_splits[new_leaf_id] = candidate_splits

                new_leaf_id = copy(tuple([1]))
                indexs = split[1]["indexs"]
                candidate_splits = get_candidate_splits(indexs)
                leaf_id_candidate_splits[new_leaf_id] = candidate_splits
            return copy(leaf_id_candidate_splits)

        def build_tree(tree):
            old_tree = copy(tree)
            old_leaf_id_candidate_splits = copy(get_leaf_id_candidate_splits(old_tree))
            old_best = copy(get_best_leaf_id_split_feature_value_score(old_tree, old_leaf_id_candidate_splits))
            score = old_best[-1]
            fit_flag = 0
            #n_splits = -np.inf
            while score != -np.inf:
                new_tree = copy(get_new_tree_with_add_best(old_tree, old_best))
                new_leaf_id_candidate_splits = copy(update_leaf_id_candidate_splits(old_leaf_id_candidate_splits, old_best))
                new_best = copy(get_best_leaf_id_split_feature_value_score(new_tree, new_leaf_id_candidate_splits))
                score = new_best[-1]
                ############################
                #########verbose############
                #n_splits += 1
                #sys.stdout.write("Split " + str(n_splits) + ", Score " + str(round(score, 4))+"\t\r")
                #sys.stdout.flush()
                ############################
                old_tree = copy(new_tree)
                old_leaf_id_candidate_splits = copy(new_leaf_id_candidate_splits)
                old_best = copy(new_best)
                fit_flag = 1
            if not fit_flag:
                new_tree = copy(tree)
            return new_tree
        
        self.tree = build_tree(tree)
        
    def predict_proba(self, X):
        
        def get_leaf_ids(tree, ids=[]):
            # apparently, when working with tuples
            # if you don't deepcopy, you can't recurse
            # that's 1.5 hour i'm not getting back...
            ids = copy(ids)
            tree = copy(tree)
            if "split" in tree:
                node_0 = tree[0]
                node_1 = tree[1]
                del tree
                id_0 = get_leaf_ids(node_0, copy(ids))
                del node_0
                id_1 = get_leaf_ids(node_1, copy(ids))
                del node_1
                ids += id_0
                del id_0
                ids += id_1
                del id_1
                return ids
            else:
                return [tree["id"]]
        
        #tree = copy(tree)
        tree = self.tree
        if "split" not in tree:
            probas = np.repeat(tree["probas"], len(X))
        else:
            indexs = np.array(range(len(X)))
            probas = np.repeat(1.0, len(X))
            leaf_ids = copy(get_leaf_ids(tree))
            for leaf_id in leaf_ids:
                leaf_indexs = copy(indexs)
                leaf_X = copy(X)
                leaf = copy(tree)
                for i in range(len(leaf_id)):
                    feature, value = leaf["split"]
                    path = leaf_id[i]
                    if path==0:
                        condition = leaf_X[leaf_indexs, feature]<value
                        if sum(condition)>0:
                            leaf_indexs = leaf_indexs[condition]
                    else:
                        condition = leaf_X[leaf_indexs, feature]>=value
                        if sum(condition)>0:
                            leaf_indexs = leaf_indexs[condition]
                    leaf = leaf[path]
                probas[leaf_indexs] = leaf["probas"]
        probas = probas.reshape(-1,1)
        probas = np.concatenate((1-probas, probas), axis=1)
        return probas

class RandomForestClassifier():
    
    def __init__(self,
        criterion="auc_div", n_estimators=100, n_bins=4, max_depth=3, bootstrap=False, max_features="sqrt", orthogonality=0.5, score_weight=False, random_state=42, n_jobs=-1
    ):
        self.n_jobs = n_jobs
        self.n_bins = n_bins
        self.criterion = criterion
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.max_features = max_features
        self.random_state = random_state
        self.score_weight = score_weight
        self.orthogonality = orthogonality
        self.trees = [DecisionTreeClassifier(
            n_bins = n_bins,
            criterion = criterion,
            max_depth = max_depth,
            bootstrap = bootstrap,
            max_features = max_features,
            score_weight = score_weight,
            random_state = random_state+i,
            orthogonality = orthogonality,
        )
        for i in range(n_estimators)]
        
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
            for dt in dt_batch:
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
        
    def predict_proba(self, X):
        def predict_proba_parallel(tree, X):
            return tree.predict_proba(X)[:,1]
        
        # Predicting
        y_preds = Parallel(n_jobs=self.n_jobs)(
            delayed(predict_proba_parallel)(
                tree, X
            ) for tree in self.trees
        )
        probas = np.mean(y_preds, axis=0).reshape(-1,1)
        probas = np.concatenate((1-probas, probas), axis=1)
        return probas

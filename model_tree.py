# decision tree

from math import *
import random
import numpy as np
from datetime import datetime as dt

rare_browsers = ["Opera Mobile", "SlimBrowser", "OmniWeb", "TheWorld Browser",
                 "Crazy Browser", "Comodo Dragon", "Flock", "PS Vita browser",
                 "NetNewsWire", "Outlook 2007", "Palm Pre web browser", "Epic",
                 "IceDragon", "Google Earth", "Kindle Browser", "Googlebot",
                 "Arora", "Conkeror", "Stainless"]

countries = ["NDF", "US", "other", "FR", "IT", "GB",
             "ES", "CA", "DE", "NL", "AU", "PT"]

def read_csv(fname = "train_users_2.csv", header = True):
    with open(fname, mode = "r") as f:
        head = None
        data = []
        if header:
            line = f.readline()
            head = line.strip().split(",")
        while True:
            line = f.readline()
            if line == "":
                break
            data.append(line.strip().split(","))
    return head, data

def flst(itr):
    # get frequency list
    f = {}
    for elem in itr:
        f[elem] = f.get(elem, 0) + 1
    return f

class Tree:
    def __init__(self):
        self.model = None
        pass

    def gini(self, y):
        # gini impurity = sum f_i (1 - f_i)
        t = len(y)
        f = flst(y)
        return sum([(v / t) * (1 - v / t) for v in f.values()])

    def info(self, y):
        # info gain = -sum f_i log_2 (f_i)
        t = len(y)
        f = flst(y)
        return -sum([(v / t) * log(v / t, 2) for v in f.values()])

    def fit(self, x, y, crit = "gini", max_depth = None, min_split = 2,
            quiet = True):
        # crit = "gini" | "info"
        if crit == "gini":
            metric = self.gini
        elif crit == "info":
            metric = self.info
        else:
            print("unrecognized criterion: " + str(crit))
            return

        # preconditions
        if (len(x) != len(y)):
            print("mismatch between number of data and labels")
            return
        if min_split < 2:
            print("min split should be at least 2")
            return

        def fit_(x, y, depth = 1, used = []):
            # model: [(ind, val, num, [model]), ...] | [(lbl)]
            model = []

            val = metric(y)

            # construct a leaf node, in certain conditions
            if (max_depth is not None and depth >= max_depth) or \
               (len(y) < min_split) or \
               (len(used) == len(x[0])) or \
               (val == 0):
                f = flst(y)
                for key in f:
                    f[key] = f[key] / len(y)
                #lbl = max(f, key = lambda v: f[v])
                return len(y), [(f, )]

            # otherwise, construct a split node
            # first select best feature to split on
            opt_val = 10 ** 10
            opt_ind = -1
            opt_lst = []
            t = len(x)
            for ind in range(len(x[0])):
                # skip attributes already used higher in the tree
                if ind in used:
                    continue

                # calculate impurity value, after split
                cur_val = 0
                attr_vals = flst(map(lambda i: i[ind], x))
                for feat in attr_vals:
                    freq = attr_vals[feat] / t
                    y2 = [lbl for i, lbl in enumerate(y) if x[i][ind] == feat]
                    cur_val += freq * metric(y2)

                # pick attribute w/ max reduction in impurity
                if (val - cur_val > val - opt_val):
                    opt_val = cur_val
                    opt_ind = ind
                    opt_lst = attr_vals

            # recursively fit the tree
            quiet or print(" " * depth + "split on: " + str(opt_ind))
            tot = 0
            for feat in opt_lst:
                x2 = [dat for dat, lbl in zip(x, y) if dat[opt_ind] == feat]
                y2 = [lbl for dat, lbl in zip(x, y) if dat[opt_ind] == feat]
                num, model_ = fit_(x2, y2, depth + 1, used + [opt_ind])
                model.append((opt_ind, feat, num, model_))
                tot += num
            return tot, model
            
        _, self.model = fit_(x, y)

    def proba(self, x):
        # return single prediction for each input element
        def predict_(model, elem):
            if len(model) == 1 and len(model[0]) == 1:
                return model[0][0]
            else:
                for ind, val, _, m2 in model:
                    if elem[ind] == val:
                        return predict_(m2, elem)
                ind = model[0][0]
                #print("no val found: " + str(ind) + " : " + str(elem[ind]))
                m2 = sorted(model, key = lambda m: m[2], reverse = True)[0][3]
                return predict_(m2, elem)
        
        predictions = []
        for elem in x:
            predictions.append(predict_(self.model, elem))
        return predictions

    def predict(self, x, k = 1):
        prob = self.proba(x)
        predictions = []
        for votes in prob:
            order = sorted(votes, key = lambda v: votes[v] \
                           if v != "" else -1, reverse = True)
            majority = order[: k]
            predictions.append(majority)
        return predictions

    def display(self, head = None):
        def display_(model, indent = ""):
            if len(model) == 1 and len(model[0]) == 1:
                print(indent + str(model[0][0]))
            else:
                m = sorted(model, key = lambda v: v[1])
                for a, b, _, m2 in m:
                    s = str(a) if head is None else head[a]
                    print(indent + s + " = " + str(b))
                    display_(m2, indent + "  ")

        display_(self.model)

class Forest:
    def __init__(self):
        self.trees = None

    def fit(self, x, y, n_trees = 10, crit = "gini", max_depth = None,
            min_split = 2, sample = 0.5, seed = 0, quiet = True):
        # list of tree objects
        self.trees = []

        n = len(x)
        data = [x[i] + [y[i]] for i in range(n)]
        random.seed(seed)
        for i in range(n_trees):
            random.shuffle(data)
            x_ = [d[: -1] for d in data[: int(n * sample)]]
            y_ = [d[-1] for d in data[: int(n * sample)]]
            t = Tree()
            t.fit(x_, y_, crit = crit, max_depth = max_depth,
                  min_split = min_split, quiet = quiet)
            self.trees.append(t)

    def predict(self, x, k = 1):
        # return list of predictions, in descending order, for each input
        predictions = []
        for elem in x:
            votes = {}
            for tree in self.trees:
                prob = tree.proba([elem])[0]
                for p in prob:
                    votes[p] = votes.get(p, 0) + prob[p]
            order = sorted(votes, key = lambda v: votes[v] \
                           if v != "" else -1, reverse = True)
            majority = order[: k]
            predictions.append(majority)
        return predictions

class AdaBoost:
    def __init__(self):
        self.trees = None
        self.alphas = None

    def fit(self, x, y, n_trees = 10, crit = "gini", max_depth = None,
            min_split = 2, sample = 0.5, seed = 0, quiet = True):
        # list of tree objects
        self.trees = []
        self.alphas = []

        # initialize weights as uniform distribution
        global w, e, a, z, p, r, t
        n = len(x)
        w = [1 / n for i in range(n)]
        data = np.array([x[i] + [y[i]] for i in range(n)])
        np.random.seed(seed)
        for i in range(n_trees):
            print("tree: " + str(i))
            idx = np.random.choice(n, int(n * sample), replace = False, p = w)
            data_ = data[idx]
            x_ = [d[: -1] for d in data_]
            y_ = [d[-1] for d in data_]
            t = Tree()
            t.fit(x_, y_, crit = crit, max_depth = max_depth,
                  min_split = min_split, quiet = quiet)
            self.trees.append(t)

            # find weighted error sum
            p = [1 if [yi] == t.predict([xi])[0] else -1
                 for xi, yi in zip(x, y)]
            r = sum([wi for pi, wi in zip(p, w) if pi == -1])
            a = 1/2 * log((1 - r) / r)
            self.alphas.append(a)

            # update weights
            z = sum([wi * (e ** (-a * pi)) for pi, wi in zip(p, w)])
            w = [wi / z * (e ** (-a * pi)) for pi, wi in zip(p, w)]

##            if val_x is not None and val_y is not None:
##                pred = self.predict(val_x, k = 5)
##                n_correct = sum([int(p[0] == y) for p, y in zip(pred, val_y)])
##                print(" val accuracy: " + str(round(n_correct / len(val_y), 3)))
##                tot = 0
##                for p, y in zip(pred, val_y):
##                    dcg = sum([((2 ** (int(p[i] == y))) - 1) / (log(i + 2, 2))
##                               for i in range(len(p))])
##                    tot += dcg
##                print(" avg dcg: " + str(round(tot / len(val_y), 5)))

    def predict(self, x, k = 1):
        # return list of predictions, in descending order
        predictions = []
        for i, elem in enumerate(x):
##            if i % (int(len(x) // 100) * 10) == 0:
##                print("pred: " + str(i))
            votes = {}
            for tree, alpha in zip(self.trees, self.alphas):
                prob = tree.proba([elem])[0]
                for p in prob:
                    votes[p] = votes.get(p, 0) + alpha * prob[p]
            order = sorted(votes, key = lambda v: votes[v] \
                           if v != "" else -1, reverse = True)
            majority = order[: k]
            predictions.append(majority)
        return predictions
        
def classify(validate = True):
    global head, x, y, tree, pred, forest
    
    print("reading training data")
    head, data = read_csv()

    print("preprocessing")
    def proc(d):
        # normalize age
        age = d[5]
        if age == "" or float(age) < 12 or float(age) > 99:
            age = -1
        else:
            a = int(float(age) // 5)
            age = a * 5
            age2 = (a + 1) * 5 - 1
            age = str(age) + "-" + str(age2)
        d[5] = str(age)

        # normalize browser
        browser = d[14]
        browser = "Other" if browser in rare_browsers else browser
        d[14] = browser
        
        return d[4 : 14 + 1]

    y = [d[-1] for d in data]
    x = [proc(d) for d in data]
    head = head[4 : 14 + 1]

    print("fitting model")
    if validate:
        split = int(0.7 * len(x))
        trn_x = x[: split]
        trn_y = y[: split]
        tst_x = x[split :]
        tst_y = y[split :]

    ##    tree = Tree()
    ##    tree.fit(trn_x, trn_y, crit = "gini", max_depth = 3, min_split = 8,
    ##             quiet = True)
    ##    forest = Forest()
    ##    forest.fit(trn_x, trn_y, n_trees = 128, max_depth = 4, min_split = 6,
    ##               quiet = True)
        ada = AdaBoost()
        ada.fit(trn_x, trn_y, n_trees = 32, max_depth = 3, min_split = 2,
                sample = 0.8, seed = 1, quiet = True)
        clf = ada

        #tree.display(head)

        print("evaluating model")
        pred = clf.predict(tst_x, k = 5)
        n_correct = sum([int(p[0] == y) for p, y in zip(pred, tst_y)])
        print(" accuracy: " + str(round(n_correct / len(tst_y), 3)))
        tot = 0
        for p, y in zip(pred, tst_y):
            dcg = sum([((2 ** (int(p[i] == y))) - 1) / (log(i + 2, 2))
                       for i in range(len(p))])
            tot += dcg
        print(" avg dcg: " + str(round(tot / len(tst_y), 5)))
    else:
##        split = int(0.85 * len(x))
##        trn_x = x[: split]
##        trn_y = y[: split]
##        val_x = x[split :]
##        val_y = y[split :]
        
##        ada = AdaBoost()
##        ada.fit(x, y, n_trees = 32, max_depth = 3, min_split = 2,
##                sample = 0.8, seed = 1, quiet = True)
        forest = Forest()
        forest.fit(x, y, n_trees = 32, max_depth = 3, min_split = 2,
                   sample = 0.95, seed = 1, quiet = True)
        clf = forest

        print("writing predictions")
        _, tst = read_csv("test_users.csv")
        ids = [t[0] for t in tst]
        tst = [proc(t) for t in tst]
        pred = clf.predict(tst, k = 5)
        with open("submission.csv", mode = "w") as f:
            f.write("id,country\n")
            for i, p in zip(ids, pred):
                for dst in p:
                    if dst != "":
                        out = f.write(str(i) + "," + str(dst) + "\n")
        
    return clf

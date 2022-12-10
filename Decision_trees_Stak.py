# -*- coding: utf-8 -*-
"""
Created on Sat Dec  1 16:53:51 2022

@author: Shubham Tak

ASU Id: 1223415479

"""

import numpy as np
from collections import Counter
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
# from sklearn import tree


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):  # * To mention Node everytime I pass value to leaf node
        self.feature = feature                                                  # Feature on which divided
        self.threshold = threshold                                              # Thresh to divide                                        
        self.left = left                                                        # Left node/tree
        self.right = right                                                      # Right node/tree                    
        self.value = value                                                      # Value if leaf node
        
    def check_leaf_node(self):                                                  # Checking for leaf node
        return self.value is not None


class DecisionTree:
    def __init__(self, min_sam_split=2, max_depth=100, n_features=None):
        self.min_sam_split=min_sam_split                                        # Stopping values
        self.max_depth=max_depth
        self.n_features=n_features                                              # To not to use all the features, to help with RF                          
        self.root=None

    def fit(self, X, y):
        
        if not self.n_features:
            
            self.n_features = X.shape[1]                                        # If min features not given, all to be used
        else:
            self.n_features = min(X.shape[1],self.n_features)                   # If given, use the given one    
        
        self.root = self.make_tree(X, y)
            
    
    def make_tree(self, X, y, depth=0):
        
        total_samples, total_features = X.shape
        total_labels = len(np.unique(y))

        
        if (depth  >= self.max_depth or total_labels == 1 or total_samples < self.min_sam_split):       # stopping criteria of tree
            
            counter = Counter(y)
            val = counter.most_common(1)[0][0]
            leaf_value = val
            return Node(value=leaf_value)

        feat_indices = np.random.choice(total_features, self.n_features, replace=False)        # To avoid duplication of features

        
        best_feature, best_thresh = self.optimum_split(X, y, feat_indices)                     # Getting the best split

        
        
        left_child = np.argwhere(X[:, best_feature] <= best_thresh).flatten()                  # Argwhere Gives indices where the condition is met, flattening to make single list
        right_child = np.argwhere(X[:, best_feature] > best_thresh).flatten()                  # create child nodes
        
        left = self.make_tree(X[left_child, :], y[left_child], depth+1)
        right = self.make_tree(X[right_child, :], y[right_child], depth+1)
        return Node(best_feature, best_thresh, left, right)


     
    def optimum_split(self, X, y, feat_indices):
        best_gain = -1
        split_index, split_threshold = None, None

        for idx in feat_indices:
            X_column = X[:, idx]                                                                # Column for that feature
            thresholds = np.unique(X_column)

            for thr in thresholds:
                
                gain = self.info_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_index = idx
                    split_threshold = thr

        return split_index, split_threshold



    def info_gain(self, y, X_column, threshold):
        
        parent_entropy = self.entropy(y)
       
        left_child = np.argwhere(X_column <= threshold).flatten()                  # Gives indices where the condition is met, flattening to make single list
        right_child = np.argwhere(X_column > threshold).flatten()
        
        if len(left_child) == 0 or len(right_child) == 0:
            return 0
        
                                                                                            # weighted avg. entropy 
        n_ysamples = len(y)
        n_left_child, n_right_child = len(left_child), len(right_child)
        
        ent_left_child, ent_right_child = self.entropy(y[left_child]), self.entropy(y[right_child])
        child_entropy = (n_left_child/n_ysamples) * ent_left_child + (n_right_child/n_ysamples) * ent_right_child

                                                                                            # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    # def gini(slef,x):
        
    #     no_occur = np.bincount(y)                                               # Gives no of occurances of lables
    #     prob_x = no_occur / len(y)
    #     for p in prob_x:
    #         if p>0:
    #             imp= 1-np.sum(p**2)
    #     return imp    
    
    
    def entropy(self, y):
        no_occur = np.bincount(y)                                               # Gives no of occurances of lables
        prob_x = no_occur / len(y)
        ent = 0
        for p in prob_x:
            if p>0:
                a = np.sum([p * np.log(p)])
                ent+=a   
                
        return -ent


    def predict(self, X):
       
      # print(np.array([self.tree_traversal(x, self.root) for x in X]))
        return np.array([self.tree_traversal(x, self.root) for x in X])

    def tree_traversal(self, x, node):
        if node.check_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.tree_traversal(x, node.left)
        return self.tree_traversal(x, node.right)


class RandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_sam_split=2, n_features=None):
        self.n_trees = n_trees
        self.max_depth=max_depth
        self.min_sam_split=min_sam_split
        self.n_features=n_features
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(max_depth=self.max_depth, min_sam_split=self.min_sam_split, n_features=self.n_features)
            X_sample, y_sample = self.bootstrapping(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def bootstrapping(self, X, y):
        total_samples = X.shape[0]
        indices = np.random.choice(total_samples, total_samples, replace=True)
        return X[indices], y[indices]

    def most_common_label(self, y):
        counter = Counter(y)
        val = counter.most_common(1)[0][0]
        return val

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(predictions, 0, 1)        
        predictions = np.array([self.most_common_label(pred) for pred in tree_preds])
     
        return predictions   


###############################################################
##################    Breast Cancer Data     #################
###############################################################


data = datasets.load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


###############################################################
##################     Car Data               #################
###############################################################

# df = pd.read_csv(r'C:\Users\shubh\OneDrive\Desktop\Study\Stat ML\Proj\data\car_evaluation.csv')

# data_columns = ["buying", "maint", "doors", "persons", "lug_boot", "safely", "sale_condition"]

# col = df.columns
# res = dict()
# for i in range(0,len(col)):
#     res[col[i]] = data_columns[i]

# df = df.rename(columns=res)

# le = preprocessing.LabelEncoder()
# df_label_encoded = df.copy()

# df_label_encoded["buying"] = le.fit_transform(df["buying"])
# df_label_encoded["maint"] = le.fit_transform(df["maint"])
# df_label_encoded["doors"] = le.fit_transform(df["doors"])
# df_label_encoded["persons"] = le.fit_transform(df["persons"])
# df_label_encoded["lug_boot"] = le.fit_transform(df["lug_boot"])
# df_label_encoded["safely"] = le.fit_transform(df["safely"])
# df_label_encoded["sale_condition"] = le.fit_transform(df["sale_condition"])

# valarr = df_label_encoded.values
# X, y = valarr[:,:-1], valarr[:,-1]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)



###############################################################
##################     Custom            ######################
###############################################################



# d = {'Age': [ "lt40", "lt30", "bw30-40", "gt40", "lt30", "gt40", "bw30-40", "lt30", "lt30", "gt40" ],
#       'Color-cloth': ["Red", "Yellow", "Blue", "Red", "White", "Red", "Blue", "Yellow", "Yellow", "White"],
#       'Income': [ 'High', 'High', 'High', 'Medium', 'Low', 'Low', 'Low', 'Medium', 'Low', 'Medium' ],
#       'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No'],
#       'Buy-Computer': ['No', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'No']}

# df = pd.DataFrame(data=d)
# columns = df.columns
# df

# le = preprocessing.LabelEncoder()
# df_label_encoded = df.copy()

# df_label_encoded["Age"] = le.fit_transform(df["Age"])
# df_label_encoded["Color-cloth"] = le.fit_transform(df["Color-cloth"])
# df_label_encoded["Income"] = le.fit_transform(df["Income"])
# df_label_encoded["Student"] = le.fit_transform(df["Student"])
# df_label_encoded["Buy-Computer"] = le.fit_transform(df["Buy-Computer"])

# X, y = df_label_encoded.values[:, :-1], df_label_encoded.values[:, -1]

# # Y = y.reshape(-1, 1)
# # Y.shape

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)



     



##########################################
# DT with entropy and IG                 #
##########################################

c_DT = DecisionTree(max_depth=10)
c_DT.fit(X_train, y_train)
predictions_DT = c_DT.predict(X_test)
acc_dt = round(accuracy_score(y_test, predictions_DT),10)
print("DT acc:",acc_dt) 

# fig = plt.figure(figsize=(25,20))
# _ = tree.plot_tree(c_DT, filled=True)
# plt.show()


# plt.figure("DT", figsize=[40,40])
# plot_tree(c_DT, fontsize=20,filled=True)
# plt.tight_layout()
# plt.show()

##########################################
# RF                                     #
##########################################


c_RF = RandomForest(n_trees=10)
c_RF.fit(X_train, y_train)
predictions_RF = c_RF.predict(X_test)
acc_rf =  round(accuracy_score(y_test, predictions_RF),10)
print("RF acc:",acc_rf)

##########################################
# CART                                   #
##########################################


c_cart = DecisionTreeClassifier()   
# print(c_cart.get_params())                                            # By default criterail is gini, so CART
c_cart.fit(X_train, y_train)
predictions_cart = c_cart.predict(X_test)
acc_cart =  round(accuracy_score(y_test, predictions_cart),10)
print("CART acc:",acc_cart)

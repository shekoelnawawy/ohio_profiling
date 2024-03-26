import pandas as pd
import joblib
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

feature_set = 'Comprehensive' #'Minimal

features_2018 = joblib.load('/Data/ExtractedFeatures/2018/'+feature_set+'.pkl')
features_2020 = joblib.load('/Data/ExtractedFeatures/2020/'+feature_set+'.pkl')

features = pd.concat([features_2018, features_2020])

standardize = False

# Using DataFrame.insert() to add a column
features.insert(len(features.columns), "Vulnerability", ['High', 'High', 'High', 'High', 'High', 'Low', 'High', 'Low', 'Low', 'High', 'High', 'High'], True)
column_names = features.columns
if standardize:
    data = StandardScaler().fit_transform(features.iloc[:, :-1])
else:
    data = np.array(features.iloc[:,:-1])

vulnerability = np.array(features.iloc[:,-1])



seeds = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000, 2000, 3000, 4000, 5000, 10000, 20000, 30000, 40000, 50000]
for seed in seeds:
    clf = tree.DecisionTreeClassifier(random_state=seed)
    clf = clf.fit(data, vulnerability)
    tree.plot_tree(clf, feature_names=column_names[:-1], filled=True, class_names=['High', 'Low'], rounded = True)
    # plt.show()
    plt.savefig('/Trees/tree_'+str(seed), dpi=300)
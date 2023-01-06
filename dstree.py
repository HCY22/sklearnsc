#!/usr/bin/python3
# usage : dstree.py [option] file
# to do : training decision tree data without test data

import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import graphviz
import os
from optparse import OptionParser

# set parameter
parser = OptionParser()
parser.add_option( '-f', '--features', type='string', dest='features', default='' )
parser.add_option( '-t', '--target', type='string', dest='target', default='' )
parser.add_option( '-d', '--depth',  type='int', dest='depth', default = 3)

(opt,args) = parser.parse_args()

fn = os.path.basename(args[0])
df = pd.read_csv(args[0], sep = "\t")

# select feature and target
features = opt.features.split(',')
X = df[features]
y = df[opt.target]

# training data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
model = DecisionTreeClassifier(max_depth=opt.depth)
model = model.fit(X, y)
print ("model score : " + str(model.score(X, y)))

#export_graphviz(model, out_file=fn + '.dot', feature_names=features, class_names=y.unique())

dot_data = tree.export_graphviz(model, out_file=None,
        feature_names=features,
        class_names=y.unique(),
        filled=True, rounded=True, special_characters=True)

graph = graphviz.Source(dot_data, format='png')
graph.render(fn)

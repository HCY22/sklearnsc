#!/usr/bin/python3
# usage : dstree.py [option] file
# to do : training decision tree data without test data

from sklearn.cluster import KMeans
from optparse import OptionParser
import matplotlib.pyplot as plt
import pandas as pd
import os

# set parameter
parser = OptionParser()
parser.add_option( '-n', '--clusters', type='int', dest='clusters', default='4' )
parser.add_option( '-t', '--title', type='string', dest='title', default='')
parser.add_option( '-l', '--label', action = 'store_true', dest = 'label' )

(opt,args) = parser.parse_args()

fn = os.path.basename(args[0])
df = pd.read_csv(args[0], sep = "\t", index_col = 0)

# KMeans
kmeans = KMeans(n_clusters= opt.clusters)
kmeans.fit(df)
y_kmeans = kmeans.predict(df)

# plot
plt.scatter(df.iloc[:,0], df.iloc[:,1], c = y_kmeans, s = 50, cmap='viridis')

if opt.title:
    plt.title( opt.title )

if opt.label:
    x = df.iloc[:,0]
    y = df.iloc[:,1]
    l = df.index.to_numpy()
    for i, t in enumerate(l):
        plt.annotate( t, (x[i], y[i]) )

plt.savefig(args[0] + '.kmeans.png')

# group txt
df['kmeans'] = y_kmeans.tolist()
df.to_csv(args[0] + '.kmeans.tsv', sep="\t")

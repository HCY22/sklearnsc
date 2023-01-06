#!/usr/bin/python3
# usage : tSNE.py [option] data.txt
# note  : This assumes input data as n points in m dimensional space,
#       : corresponding to n rows of m-coordinates in a matrix form.

import sys
from optparse import OptionParser
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# set parameter

parser = OptionParser()
parser.add_option( '-t', '--title', type='string', dest='title', default='')
parser.add_option( '-l', '--label', action = 'store_true', dest = 'label' )
parser.add_option( '-n', '--comp', type = 'int', dest = 'comp', default = 2)
parser.add_option( '-p', '--perplexity', type = 'float', dest = 'perplexity', default = 30)

(opt,args)=parser.parse_args()


# load data points

d = pd.read_csv( args[0], delimiter = '\t', index_col = 0 )
X = d.to_numpy()

tsne = TSNE( n_components = opt.comp, perplexity = opt.perplexity )
X_r = tsne.fit_transform(X)

x = X_r[:,0]
y = X_r[:,1]
l = d.index.to_numpy()


# n_components tsv
hd = np.array(['PC' + str(a) for a in np.arange(X_r.shape[1])+1])
#np.savetxt(args[0] + '.tsne.tsv' , X_r, delimiter='\t', fmt='%f', header = "\t".join(hd))

# add colname and save
np.savetxt(args[0] + '.tsne.tsv', np.append(np.reshape(l, (-1,1)) , X_r, axis=1), delimiter='\t', fmt="%s", header = "tsne\t" + "\t".join(hd))

# scatter plot

plt.plot( x, y, 'o' )

#plt.xlabel( 'PC1 (' + str(vr[0]) + '%)' )
#plt.ylabel( 'PC2 (' + str(vr[1]) + '%)' )

if opt.title:
    plt.title( opt.title )

if opt.label:
    for i, t in enumerate(l):
        plt.annotate( t, (x[i], y[i]) )
    
plt.savefig( args[0] + '.tsne.png' )

import os
import pandas as pd
import numpy as np

import matplotlib


from matplotlib import pylab as plt
import cPickle as pickle
import itertools
from matplotlib.path import Path
import matplotlib.patches as patches



from fuel.datasets.hdf5 import H5PYDataset
import fuel
from sphinx.quickstart import mkdir_p
datasource = 'handwriting'
datasource_dir = os.path.join(fuel.config.data_path[0], datasource)
datasource_fname = os.path.join(datasource_dir , datasource+'.hdf5')
mkdir_p(datasource_dir)
print datasource_dir, datasource_fname



import xml.etree.ElementTree as ET

def readPoints(fname):
    tree = ET.parse(fname)
    root = tree.getroot()
    points = []
    sets = 0
    for strokeSet in root.iter('StrokeSet'):
        assert sets == 0
        sets += 1
        for stroke in strokeSet.iter('Stroke'):
            start = 1
            for point in stroke.iter('Point'):
                points.append([int(point.attrib['x']), int(point.attrib['y']), start])
                start = 0
    points = np.array(points)
    points = np.hstack((points[1:,:2] - points[:-1,:2], points[1:,2,None]))  # keep deltas
    return points

from sketch import drawpoints
plt.figure(figsize=(10,10))
count = 1
M = 4
root = os.path.join(datasource_dir , "lineStrokes")
for dirpath, dnames, fnames in os.walk(root):
    for f in fnames:
        if count > M: break
        if f.endswith(".xml"):
            points = readPoints(os.path.join(dirpath,f))
            plt.subplot(4,1,count)
            count += 1
            drawpoints(points)
    if count > M:
        break



all_points = []
for dirpath, dnames, fnames in os.walk(root):
    for f in fnames:
        if f.endswith(".xml"):
            points = readPoints(os.path.join(dirpath,f))
            all_points.append(points)

len(all_points)
lens = map(len, all_points)
min(lens),max(lens)
plt.hist(lens,bins=50)
plt.xlabel('#points')
plt.ylabel('#sketches')
plt.title('Distribution of number of points in a sketch');
short_points_idx = filter(lambda i: len(all_points[i])<1200, range(len(all_points)))
N=len(short_points_idx)
N

xys = np.vstack([all_points[i][:,:2] for i in short_points_idx])
xys.shape

m = xys.mean(axis=0)
s = xys.std(axis=0)
m,s

X = np.empty((N,1200,3))
for i,j in enumerate(short_points_idx):
    p = all_points[j]
    n = len(p)
    X[i,:n,:2] = p[:,:2]/s
    X[i,n:,:2] = 0
    X[i,:n,2] =  p[:,2]
    X[i,n:,2] = 1
    
X.shape
import random
X = X[random.sample(xrange(N),N)]
import h5py
fp = h5py.File(datasource_fname, mode='w')
image_features = fp.create_dataset('features', X.shape, dtype='float32')
image_features[...] = X
N_train = int(9*len(X)/10)
N_train
split_dict = {
    'train': {'features': (0, N_train)},
    'test': {'features': (N_train, N)}
}
fp.attrs['split'] = H5PYDataset.create_split_array(split_dict)
fp.flush()
fp.close()




                
    





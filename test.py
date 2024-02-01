import torch
import numpy as np
from scipy.spatial.distance import mahalanobis

dist_list = []

dist = 1
H,W = 2,2
B = 3

dist_list= [1,1,1,1,1,1,1,1,1,1,1,1]
d = np.array(dist_list)#.transpose(1, 0) # [1,12
d = d.transpose(1, 0)
print(d)
#dist_list = d.reshape(B, H, W)


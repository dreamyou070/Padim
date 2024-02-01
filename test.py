import torch
import numpy as np
from scipy.spatial.distance import mahalanobis

embedding_vectors = torch.randn((2,4))
mean = torch.mean(embedding_vectors, dim=0).numpy()
B, C = embedding_vectors.size()
covariance = np.cov(embedding_vectors.numpy(), rowvar=False)

print(f'embedding_vectors: {embedding_vectors}')
print(f'mean: {mean}')
print(f'covariance: {covariance}')
# calculate distance matrix

sample = torch.randn(4)
dist = mahalanobis(sample, mean, covariance)

print(f'dist : {dist}')
d = torch.tensor(dist)
print(f'd : {d}')
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

test_embedding_vectors = torch.randn((3,4))
B, C = test_embedding_vectors.size()
embedding_vectors = test_embedding_vectors.numpy()

dist_list = []

dist = [mahalanobis(sample, mean, covariance) for sample in embedding_vectors]
dist_list.append(dist)

dist_list = np.array(dist_list).transpose(1, 0).reshape(B, H, W)
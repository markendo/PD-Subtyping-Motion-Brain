from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve, auc

# Within-cluster sum of squares metric. In the paper, denoted as σ
def wcss_score(labels, features):
    groups = defaultdict(list)
    centroids = {}
    for i, label in enumerate(labels):
        groups[label].append(i)
    
    for group in groups:
        centroids[group] = np.mean(features[groups[group],:], axis=0)
    
    score = 0
    for group, centroid in centroids.items():
        for member in groups[group]:
            score += np.linalg.norm(features[member,:] - centroid) ** 2
            
    return score

# from A Similarity Measure for Clustering and Its Applications
# https://rambasnet.github.io/pdfs/ClusteringSimilarityAndItsApplications.pdf
def cluster_sim_score(labels1, labels2):
    groups1 = defaultdict(set)
    groups2 = defaultdict(set)
    for i, label in enumerate(labels1):
        groups1[label].add(i)
    for i, label in enumerate(labels2):
        groups2[label].add(i)
    S = np.zeros((len(groups1), len(groups2)))
    for index_i, i in enumerate(groups1):
        c_i = groups1[i]
        for index_j, j in enumerate(groups2):
            d_j = groups2[j]
            S[index_i][index_j] = len(c_i & d_j) / len(c_i | d_j)
    return np.sum(S) / max(len(groups1), len(groups2))
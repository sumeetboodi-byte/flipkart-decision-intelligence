import csv
import math
import os
import numpy as np


X_raw, y_raw, val_raw = [], [], []
with open('data/records.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        X_raw.append([float(row[i]) for i in range(1, 6)]) # X1 to X5
        y_raw.append(float(row[8]))
        val_raw.append(float(row[9]))

X = np.array(X_raw)
y = np.array(y_raw)
val = np.array(val_raw)
n = len(X)

mean_X = np.mean(X, axis=0)
sd_X = np.std(X, axis=0, ddof=1)
X_std = (X - mean_X) / sd_X


def run_kmeans(K):
    
    indices = [i * (800 // K) for i in range(K)]
    centroids = X_std[indices].copy()
    
    while True:
        
        distances = np.sum((X_std[:, np.newaxis, :] - centroids[np.newaxis, :, :])**2, axis=2)
        labels = np.argmin(distances, axis=1)
        
        
        new_centroids = np.copy(centroids)
        for k in range(K):
            members = X_std[labels == k]
            if len(members) > 0:
                new_centroids[k] = np.mean(members, axis=0)
                
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
        
    
    wcss = 0.0
    for i in range(len(X_std)):
        wcss += np.sum((X_std[i] - centroids[labels[i]])**2)
        
    return labels, wcss

wcss_values = {}
for K in range(2, 7):
    _, wcss_values[K] = run_kmeans(K)


elbow_k = None
for K in [3, 4, 5]:
    drop = wcss_values[K] - wcss_values[K+1]
    prev_drop = wcss_values[K-1] - wcss_values[K]
    if drop < 0.5 * prev_drop:
        elbow_k = K
        break
if elbow_k is None:
    elbow_k = 6 


labels_3, _ = run_kmeans(3)


cluster_x3_means = [np.mean(X[labels_3 == k, 2]) for k in range(3)]
order = np.argsort(cluster_x3_means) # Ascending order
segment_names = ['A', 'B', 'C']
segment_labels = np.zeros(n)
for new_label, old_label in enumerate(order):
    segment_labels[labels_3 == old_label] = new_label

seg_data = []
for i, name in enumerate(segment_names):
    mask = segment_labels == i
    size = int(np.sum(mask))
    m_X1 = np.mean(X[mask, 0])
    m_X2 = np.mean(X[mask, 1])
    m_X3 = np.mean(X[mask, 2])
    m_X4 = np.mean(X[mask, 3])
    m_X5 = np.mean(X[mask, 4])
    outcome_rate = np.mean(y[mask])
    mean_val = np.mean(val[mask])
    seg_data.append([name, size, m_X1, m_X2, m_X3, m_X4, m_X5, outcome_rate, mean_val])


dissim = {
    'E1': [0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
    'E2': [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
    'E3': [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
    'E4': [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
    'E5': [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
    'E6': [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]
}
names = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']

def get_dist(c1, c2):
    d = []
    for m1 in c1.split('+'):
        for m2 in c2.split('+'):
            i1, i2 = names.index(m1), names.index(m2)
            d.append(dissim[m1][i2])
    return d

def hierarchical(linkage):
    clusters = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6']
    merges = []
    heights = []
    for step in range(5):
        best_d = float('inf')
        best_pair = None
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dists = get_dist(clusters[i], clusters[j])
                if linkage == 'single': d = min(dists)
                elif linkage == 'complete': d = max(dists)
                else: d = sum(dists) / len(dists)
                
                if d < best_d:
                    best_d = d
                    best_pair = (i, j)
        
        c1, c2 = clusters[best_pair[0]], clusters[best_pair[1]]
        # Format cluster names
        m1, m2 = sorted(c1.split('+')), sorted(c2.split('+'))
        new_c = '+'.join(sorted(m1 + m2))
        
        merges.append([linkage, step+1, c1, c2, best_d])
        heights.append(best_d)
        
        
        clusters.pop(max(best_pair))
        clusters.pop(min(best_pair))
        clusters.append(new_c)
        
    
    gaps = [heights[i+1] - heights[i] for i in range(4)]
    max_gap_idx = np.argmax(gaps)
    clusters_at_cut = 6 - (max_gap_idx + 1)
    
    return merges, clusters_at_cut

merges_s, cut_s = hierarchical('single')
merges_c, cut_c = hierarchical('complete')
merges_a, cut_a = hierarchical('average')


# (1) Mean vector (already zero because we standardised)
# (2) Centre data (already zero)
# (3) Covariance matrix
cov_mat = np.dot(X_std.T, X_std) / (n - 1)
# (4) Eigen-decomposition
eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
# (5) Sort descending
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
# (6) Keep top 2
W = eigenvectors[:, :2]
# (7) Project
Z = np.dot(X_std, W)


for i in range(2):
    if np.abs(W[:, i]).max() == -W[:, i].min():
        W[:, i] = -W[:, i]
        Z[:, i] = -Z[:, i]
    elif np.abs(W[:, i]).max() == W[:, i].max() and W[:, i].max() < 0:
        W[:, i] = -W[:, i]
        Z[:, i] = -Z[:, i]


expl_ratios = eigenvalues / np.sum(eigenvalues)
cum_ratios = np.cumsum(expl_ratios)
for i in range(len(cum_ratios)):
    if cum_ratios[i] >= 0.8:
        comps_80 = i + 1
        break

pc1_means = [np.mean(Z[segment_labels == i, 0]) for i in range(3)]


os.makedirs('outputs', exist_ok=True)

with open('outputs/part3_segments.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['segment', 'size', 'mean_X1', 'mean_X2', 'mean_X3', 'mean_X4', 'mean_X5', 'outcome_rate', 'mean_value'])
    for r in seg_data:
        w.writerow([r[0], r[1]] + [f"{x:.4f}" for x in r[2:]])

with open('outputs/part3_hierarchical.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['linkage', 'merge_no', 'cluster_a', 'cluster_b', 'height'])
    for r in merges_s + merges_c + merges_a:
        w.writerow([r[0], r[1], r[2], r[3], f"{r[4]:.1f}"])

with open('outputs/part3_results.txt', 'w') as f:
    for K in range(2, 7):
        f.write(f"wcss_k{K} = {wcss_values[K]:.4f}\n")
    f.write(f"elbow_k = {elbow_k}\n")
    f.write(f"size_segment_A = {seg_data[0][1]}\n")
    f.write(f"size_segment_B = {seg_data[1][1]}\n")
    f.write(f"size_segment_C = {seg_data[2][1]}\n")
    f.write(f"single_clusters_at_cut = {cut_s}\n")
    f.write(f"complete_clusters_at_cut = {cut_c}\n")
    f.write(f"average_clusters_at_cut = {cut_a}\n")
    f.write(f"eigenvalue_sum = {np.sum(eigenvalues):.4f}\n")
    for i in range(5):
        f.write(f"eigenvalue_{i+1} = {eigenvalues[i]:.4f}\n")
    for i in range(5):
        f.write(f"explained_ratio_{i+1} = {expl_ratios[i]:.4f}\n")
    for i in range(5):
        f.write(f"cumulative_ratio_{i+1} = {cum_ratios[i]:.4f}\n")
    f.write(f"components_for_80pct = {comps_80}\n")
    for i in range(5):
        f.write(f"pc1_loading_X{i+1} = {W[i, 0]:.4f}\n")
    for i in range(5):
        f.write(f"pc2_loading_X{i+1} = {W[i, 1]:.4f}\n")
    for i in range(3):
        f.write(f"row{i+1}_pc1 = {Z[i, 0]:.4f}\n")
        f.write(f"row{i+1}_pc2 = {Z[i, 1]:.4f}\n")
    f.write(f"mean_pc1_segment_A = {pc1_means[0]:.4f}\n")
    f.write(f"mean_pc1_segment_B = {pc1_means[1]:.4f}\n")
    f.write(f"mean_pc1_segment_C = {pc1_means[2]:.4f}\n")

print("Success! Part 3 outputs saved.")
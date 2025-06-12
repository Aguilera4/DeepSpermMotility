import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN

df = pd.read_csv('../results/data_sperm_tracking/sperm_tracking_11_labels.csv')


grouped = df.groupby('track_id')

def extract_features(traj):
    traj = traj.sort_values('frame_id')
    x = traj['cx'].values
    y = traj['cy'].values

    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.sqrt(dx**2 + dy**2)

    total_distance = np.sum(dist)
    net_displacement = np.linalg.norm([x[-1] - x[0], y[-1] - y[0]])
    mean_speed = np.mean(dist)
    linearity = net_displacement / total_distance if total_distance > 0 else 0
    curvature = np.mean(np.abs(np.diff(np.arctan2(dy, dx)))) if len(dx) > 1 else 0

    return [total_distance, net_displacement, mean_speed, linearity, curvature]


features = []
sperm_ids = []

for sperm_id, traj in grouped:
    if len(traj) < 5:  # skip short trajectories
        continue
    sperm_ids.append(sperm_id)
    features.append(extract_features(traj))

X = np.array(features)
X_scaled = StandardScaler().fit_transform(X)

# Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Visualization with t-SNE
X_embedded = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=labels, cmap='tab10', alpha=0.8)
plt.title("Sperm Trajectory Clustering")
plt.colorbar(scatter)
plt.show()
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('../results/data_features/dataset_30s_v2.csv')

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, init='k-means++', n_init=100, max_iter=1000, tol=0.0001, random_state=42)

df_copy = df.loc[:,  ['displacement','vcl','vsl','lin']]

df_copy['label'] = kmeans.fit_predict(df_copy)


'''df[df['cluster'] == 0].to_csv("cluster0.csv")
df[df['cluster'] == 1].to_csv("cluster1.csv")
df[df['cluster'] == 2].to_csv("cluster2.csv")
df[df['cluster'] == 3].to_csv("cluster3.csv")'''

from sklearn.decomposition import PCA

'''pca = PCA(n_components=2)
X_pca = pca.fit_transform(df)
df['pca1'] = X_pca[:,0]
df['pca2'] = X_pca[:,1]

import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(data=df,x='pca1',y='pca2',hue='cluster')
plt.show()'''

#df.to_csv("dataset_clustering_4c_22.csv", index=False)

cluster_stats = df_copy.groupby('label').agg(['mean', 'std', 'min', 'max'])
print(cluster_stats)

'''sns.pairplot(df_copy, hue='label', diag_kind='kde')
plt.show()'''

# Entrenamiento de un árbol de decisión para extraer reglas
features = ['displacement','vcl','vsl','lin']
X = df_copy[features]
y = df_copy['label']

tree = DecisionTreeClassifier(
    criterion='gini',             # Gini impurity
    max_depth=3,                  # Limit tree depth
    min_samples_split=10,         # Minimum samples to split a node
    min_samples_leaf=5,           # Minimum samples at a leaf node
    max_features='sqrt',          # Use square root of features for splits
    random_state=42,              # Ensure reproducibility
    class_weight='balanced'       # Handle class imbalance (if any)
    )
tree.fit(X, y)

# Mostrar reglas extraídas
rules = export_text(tree, feature_names=features)
print("Reglas de clasificación extraídas del árbol de decisión:")
print(rules)
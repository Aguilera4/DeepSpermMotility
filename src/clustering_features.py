import pandas as pd


df = pd.read_csv('../results/data_features_labelling_preprocessing/dataset_4c_30s_preprocessing.csv')

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=4, random_state=0)


df_copy = df.loc[:, ['vcl','vsl','linearity','vap','alh']]

df['label'] = kmeans.fit_predict(df_copy)


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

df.to_csv("dataset_clustering_4c_30s.csv", index=False)


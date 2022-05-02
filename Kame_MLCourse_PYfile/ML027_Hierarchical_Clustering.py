# %%
from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np

# 擬似データ
x1 = np.array([1.2, 0.4, 1.6, 2.2, 4.1, 5.6, 6, 6.7, 7.3, 8, 8.6])
x2 = np.array([0.3, 1.7, 7, 1, 3.1, 4.1, 2.5, 7.7, 4.2, 3.3, 7.6])

# 分かりやすいように各データに数字を振る
index = np.arange(len(x1))
X = np.array(list(zip(x1, x2)))

# 可視化
plt.plot(X[:, 0], X[:, 1], 'o')
for i in np.arange(len(x1)):
    plt.annotate(f'{i}', (x1[i], x2[i]+0.1), size=15)

# %%
Z = linkage(X, 'ward')
Z

# %%
d = dendrogram(Z)

# %%
d = dendrogram(Z, truncate_mode='lastp', p=4)

# %%
# ラベル取得
from scipy.cluster.hierarchy import fcluster
clusters = fcluster(Z, t=4, criterion='maxclust')
clusters

# %%
from matplotlib import pyplot as plt
%matplotlib inline

plt.scatter(X[:, 0], X[:, 1], c=clusters)

# %%




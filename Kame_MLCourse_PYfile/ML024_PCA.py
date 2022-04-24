# %%
# データ読み込み
import seaborn as sns
df = sns.load_dataset('iris')
df.head()

# %%
# PCAを行う準備。データを標準化する。
from sklearn.preprocessing import StandardScaler
# 特徴量と目的変数を分離
X = df.loc[:, df.columns != 'species']
y = df['species'].values
# 標準化
X = StandardScaler().fit_transform(X)

# %%
# PCAの実施
from sklearn.decomposition import PCA
# 各成分の寄与率を確認
pca = PCA(n_components=len(X[0]))
pca.fit(X)
# 寄与率
pca.explained_variance_ratio_

# %%
# 累積寄与率を計算
import numpy as np
np.cumsum(pca.explained_variance_ratio_)

# %%
import matplotlib.pyplot as plt
%matplotlib inline
 
n_components = len(X[0])
plt.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_))
plt.xticks(range(1, n_components+1))
plt.xlabel('components')
plt.xlabel('components')
plt.ylabel('cumulative explained variance')

# %%
# 寄与率、累積寄与率の分析から2次元に圧縮することとする
pca = PCA(n_components=2)
pca.fit(X)
X_pc = pca.transform(X)

# %%
# 2次元プロットする準備
import pandas as pd
# X_pcとyをconcatenateするためにyの行数をXに合わせる。行と列を入れ替え。
y = y.reshape(-1, 1)
# X_pcとyをconcatenateしたものをDataFrameにする
df_pc = pd.DataFrame(np.concatenate([X_pc, y], axis=1), columns=['first component', 'second component', 'species']).astype({'first component':float, 'second component':float})

# %%
# 2次元プロット実施
sns.scatterplot(x='first component', y='second component', hue='species', data=df_pc)

# %%




# %%
import seaborn as sns

# データの読み込み
df = sns.load_dataset('iris')
# 学習データとテストデータ作成
# 教師なし学習なので，目的変数である’species’カラムは使用しない
X = df.loc[:, df.columns!='species']
X.head()

# %%
# 教師なし学習なので，学習データとテストデータに分けず，全てのデータを使う
# (特徴量のスケールに差がある場合は標準化をしてからk-meansを行うと良い)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)

# %%
# クラスタの結果を確認
labels = kmeans.labels_
labels

# %%
# クラスタリング結果と元データXを結合したDFを作成、可視化
import pandas as pd
result_df = pd.concat([X, pd.DataFrame(labels, columns=['kmeans_result'])], axis=1)
sns.pairplot(result_df, hue='kmeans_result')

# %%
# クラスターの数を変えて損失の推移を確認
import matplotlib.pyplot as plt
losses = []
for K in range(1, 10):
    kmeans = KMeans(n_clusters=K)
    kmeans.fit(X)
    losses.append(-kmeans.score(X))
plt.plot(range(1, 10), losses)
plt.xlabel('K')
plt.ylabel('loss')

# %%

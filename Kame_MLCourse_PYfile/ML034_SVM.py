# %%
# データ準備
import pandas as pd
import seaborn as sns

# データロード
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# %%
# データセットを学習データとテストデータを7:3に分ける
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %%
# SVMは事前に標準化しておくことが推奨される
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
# 今回は描画したいのでPCAで次元削減する
# また、PCAする前には標準化が必要
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# %%
# モデル構築
"""
C : 誤差の正則化項の係数. デフォルトは1
kernel : 使用するカーネル関数を 'linear', 'poly', 'rbf', 'precomputed' から選択.
         デフォルトは 'rbf'
degree : 'poly' を選択したときのdegree d. デフォルトは3
gamma : 'polynomial' , 'rbf' , 'sigmoid' の係数γ.
        'scale' (デフォルト)を指定すると1/(n*Var(X).
        'auto' を指定すると1/n(ただしnは特徴量の数)
"""
from sklearn import svm
model = svm.SVC(kernel='linear')

# 学習
model.fit(X_train_PCA, y_train)

# %%
# 予測と評価
from sklearn.metrics import accuracy_score
# 予測
y_pred = model.predict(X_test_pca)
# 評価
accuracy_score(y_test, y_pred)

# %%
# サポートベクトルの可視化
# サポートベクトルのリスト
model.support_vectors_

# %%
len(model.support_vectors_)

# %%
from sklearn.inspection import DecisionBoundaryDisplay
import numpy as np
import matplotlib.pyplot as plt

# 決定境界の描画
DecisionBoundaryDisplay.from_estimator(model,
                                       X_train_pca,
                                       plot_method='contour',
                                       cmap=plt.cm.Paired,
                                       levels=[-1, 0, 1],
                                       alpha=0.5,
#                                        linestyles=['--', '-', '--'],
                                       xlabel='first principal component',
                                       ylabel='second principal component',
                                       )

# 学習データの描画
for i, color in zip(model.classes_, 'bry'):
    idx = np.where(y_train == i)
    plt.scatter(
            X_train_pca[idx, 0],
            X_train_pca[idx, 1],
            c=color,
            label=iris.target_names[i],
            edgecolor='black',
            s=20,
    )

# サポートベクトルの描画
plt.scatter(
    model.support_vectors_[:, 0],
    model.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors='none',
    edgecolors='k')

# %%




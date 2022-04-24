# %%
# データ読み込み
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')
mnist.data.head()

# %%
# 探索的データ探索(EDA)開始↓
# 統計値を見てみる
mnist.data.describe()

# %%
# 正解ラベルを見てみる
mnist.target

# %%
# 画像を再構成
import matplotlib.pyplot as plt
idx = 0
# loc[ラベル]で列を取得。iloc[列番号]で列を取得
# 今回idxがラベルなのでloc[idx]で列を取得
im = mnist.data.loc[idx].values.reshape(28, 28)
plt.imshow(im, 'gray')

# %%
# 学習データとテストデータを作成
# 学習データ:テストデータ = 7:3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(mnist.data, mnist.target, test_size=0.3, random_state=0)

# %%
# PCAを行う準備。データを標準化する。
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# X_trainでfitする
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# %%
# PCAの実施
from sklearn.decomposition import PCA
# 累積寄与率を指定して95%になるまでの主成分を返すようにする
pca = PCA(n_components=0.95)
pca.fit(X_train)
X_train_pc = pca.transform(X_train)
X_test_pc = pca.transform(X_test)
print(f'{X_train.shape[-1]} dimention is reduced to {X_train_pc.shape[-1]} dimention by PCA')

# %%
# PCA後のデータX_train_pcでロジスティック回帰
from sklearn.linear_model import LogisticRegression
import time
model_pca = LogisticRegression()
before = time.time()
model_pca.fit(X_train_pc, y_train)
after = time.time()
print(f'fit took {after-before:.2f}s')

# %%
# 予測
model_pca.predict(X_test_pc[0].reshape(1, -1))

# %%
# X_test_pc[0]に対応する正解ラベルを確認
y_test.iloc[0]

# %%
# PCAなしバージョンでロジスティック回帰してみる
from sklearn.linear_model import LogisticRegression
import time
model = LogisticRegression()
before = time.time()
model.fit(X_train, y_train)
after = time.time()
print(f'fit took {after-before:.2f}s')

# %%
# AUCでそれぞれのモデルを評価する
# PCAモデルと普通のモデルの2回行うので関数化する
# ROC
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
 
 
def all_roc(model, X_test, y_test):
    # クラス数
    n_classes = len(model.classes_)
    # OvRのためone-hotエンコーディング
    y_test_one_hot = label_binarize(y_test, classes=list(map(str, range(n_classes))))
    predict_proba = model.predict_proba(X_test)
    fpr = {}
    tpr = {}
    roc_auc = {}
    # 各クラス毎にROCとAUCを計算
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], predict_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # micro平均
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test_one_hot.ravel(), predict_proba.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])
    
    # macro平均
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr = mean_tpr / n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    return fpr, tpr, roc_auc

# %%
# PCAモデルと普通のモデルそれぞれに関数を適用
fpr, tpr, roc_auc = all_roc(model, X_test, y_test)
fpr_pca, tpr_pca, roc_auc_pca = all_roc(model_pca, X_test_pc, y_test)

# %%
import pandas as pd
result_df = pd.DataFrame([roc_auc, roc_auc_pca]).T.rename(columns={0:'normal', 1:'pca'})
result_df

# %%
result_df['pca - normal'] = result_df['pca'] - result_df['normal']
result_df

# %%




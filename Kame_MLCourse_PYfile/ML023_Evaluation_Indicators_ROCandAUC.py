# %%
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# データロード
df = sns.load_dataset('iris')

# 学習データとテストデータ作成
X = df.loc[:, df.columns != 'species']
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# モデル構築
model = LogisticRegression()
model.fit(X_train, y_train)

# 予測確率
y_pred_proba = model.predict_proba(X_test)

# %%
y_test[:5]

# %%
model.classes_

# %%
from sklearn.preprocessing import label_binarize
y_test_one_hot = label_binarize(y_test, classes=model.classes_)
y_test_one_hot[:5]

# %%
from sklearn.metrics import roc_curve, auc
n_classes = 3
fpr = {}
tpr = {}
roc_auc = {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred_proba[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


# %%
import matplotlib.pyplot as plt
%matplotlib inline

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'class: {i}')
plt.legend()

# %%
import numpy as np
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# %%
all_fpr

# %%
mean_tpr = np.zeros_like(all_fpr)
mean_tpr

# %%
# 各クラス毎に、all_fpr(x軸)に対するtprを補完値含めて準備
# tprの平均mean_tprを計算する
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
mean_tpr = mean_tpr / len(model.classes_)
mean_tpr

# %%
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'class: {i}')
plt.plot(fpr['macro'], tpr['macro'], label='macro')
plt.legend()

# %%
# .flatten()でもOK
# データ1に対するclass予測[0, 0, 1]は3つの予測[0], [0], [1]として格納される
fpr['micro'], tpr['micro'], _ = roc_curve(y_test_one_hot.ravel(), y_pred_proba.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# %%
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], label=f'class: {i}')
plt.plot(fpr['macro'], tpr['macro'], label='macro')
plt.plot(fpr['micro'], tpr['micro'], label='micro')
plt.legend()

# %%




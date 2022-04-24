# %%
import seaborn as sns
df = sns.load_dataset('iris')
df.head()

# %%
sns.pairplot(df, hue='species')

# %%
from sklearn.model_selection import train_test_split
X = df.loc[:, df.columns!='species']
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
print(len(X_train), len(X_test))

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(penalty='none')
model.fit(X_train, y_train)

# %%
X_test.head()

# %%
model.predict(X_test)

# %%
model.predict_proba(X_test)

# %%
model.classes_

# %%
model.intercept_

# %%
model.coef_

# %%
model.feature_names_in_

# %%
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
 
# データロード
df = sns.load_dataset('iris')
# 学習データとテストデータ作成
X = df.loc[:, df.columns!='species']
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# モデル構築
model = LogisticRegression(penalty='none')
model.fit(X_train, y_train)
# 予測(ラベル)
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

# %%
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()

# %%
# Accuracy (正解率) 判定全体(TP+FP+TN+FN)のうち、どれだけ正解(TP+TN)か。算術平均
from sklearn.metrics import accuracy_score

# accuracy
accuracy_score(y_test, y_pred)

# %%
# Precision (適合率) Positiveとしたすべての判定(TP+FP)のうち、どれだけ正解(TP)か
from sklearn.metrics import precision_score
# None
print(precision_score(y_test, y_pred, average=None))
# macro
print(precision_score(y_test, y_pred, average='macro'))
# micro -> Accuracy(算術平均)と同じ
print(precision_score(y_test, y_pred, average='micro'))

# %%
# Recall (再現率) Positiveのデータ全体(TP+FN)のうち、どれだけPositiveと正しく判定できたか(TP)
from sklearn.metrics import recall_score
# None
print(recall_score(y_test, y_pred, average=None))
# macro
print(recall_score(y_test, y_pred, average='macro'))
# micro -> Accuracy(算術平均)と同じ
print(recall_score(y_test, y_pred, average='micro'))

# %%
# Specificity (特異度) Negativeのデータ全体(TN+FP)のうち、どれだけNegativeと正しく判定できたか(TN)
# scikit-learnには実装されていないので、PositiveとNegativeのデータを入れ替え後、Recallで計算
import numpy as np
res = []
# 各クラスcについて、TrueとFalseを入れ替える (label != c)
# -> recall_scoreで各クラスcにおけるRecall(再現率)を計算
for c in model.classes_:
    res.append(recall_score(np.array(y_test)!=c, np.array(y_pred)!=c))
res
# 各クラスの平均を取ることで、macro平均を算出
np.mean(res)

# %%




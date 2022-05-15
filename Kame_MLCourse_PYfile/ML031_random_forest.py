# %%
# データ準備
import pandas as pd
import seaborn as sns

# データロード
df = sns.load_dataset('titanic')
# 欠損値drop
df = df.dropna()
# 特徴量
X = df.loc[:, (df.columns != 'survived') & (df.columns != 'alive')]
# 質的変数をダミー変数化
X = pd.get_dummies(X, drop_first=True)
# 目的変数
y = df['survived']


# %%
# モデル構築
"""
n_estimators : アンサンブルする決定木の数。
  デフォルトは100。数が多い方がいいがその分時間がかかるので注意。
max_depth : 決定木の深さ。何も指定しない( None )と最後まで分割が走ってしまう。
  時間がかかる上に過学習してしまうので注意
  (他の条件により途中で分割が終わるようなら None でもOK)
min_samples_split : intで指定した場合は分割する際に必要な最低限のデータ数。
  floatを指定すると全データ数の割合。
max_features : 決定木に使う特徴量の数。デフォルトは 'auto' で特徴量数nとすると√nになる
ccp_alpha : cost complexity pruningのalphaの値。
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
rf_model = RandomForestClassifier(random_state=0, ccp_alpha=0.02)
dt_model = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=0.02)

# %%
# 学習と評価
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=0)
rf_scores = cross_val_score(rf_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
dt_scores = cross_val_score(dt_model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)

# %%
score_df = pd.DataFrame({'random forest': rf_scores, 'decision tree': dt_scores})
sns.barplot(data=score_df)

# %%
from scipy import stats
stats.ttest_rel(score_df['random forest'], score_df['decision tree'])

# %%
# .feature_importances_ で各特徴量の重要度を確認
import matplotlib.pyplot as plt
rf_model.fit(X, y)
# plt.barh(y_label, y_value)
plt.barh(X.columns, rf_model.feature_importances_)

# %%




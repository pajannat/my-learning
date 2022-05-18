# %%
# XGBoostはscikit-learnの中に実装されていないのでインストール
%pip install xgboost

# %%
# データ準備
import pandas as pd
import seaborn as sns

# データロード
df = sns.load_dataset('titanic')
# 欠損値drop (XGBoostは欠損値を対処するアルゴリズムを含むため不要)
# df = df.dropna()
# 特徴量
X = df.loc[:, (df.columns != 'survived') & (df.columns != 'alive')]
# 質的変数をダミー変数化
X = pd.get_dummies(X, drop_first=True)
# 目的変数
y = df['survived']

# %%
# データセットを学習データとテストデータを7:3に分ける
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %%
# モデル構築
"""
n_estimators : ブースティングの回数(=学習する決定木の数)
               デフォルトは100
learning_rate : shrinkageのη
                デフォルトは0.3で少し高め(低スペックマシンでの実行を想定?)
max_depth : 決定木の最高の深さ デフォルトは6
eval_metric : ブースティング時の各イテレーション時に使う評価指標
              (特に後述するearly stoppingに使用)
              sklearn.metricsのメソッドを渡すか自作 デフォルトは 'logloss'
early_stopping_rounds : early stoppingする際の最低限のイテレーション回数
"""
# early stoppingで評価指標が下がらなくなったら自動で学習終了できる
# n_estimatorsやlearning_rateで学習が長くなる値を設定してok
from xgboost import XGBClassifier
model = XGBClassifier(early_stopping_rounds=10)

# %%
# 学習
"""
eval_set : (X, y)のリスト 
           ブースティング時の各イテレーションごとにこのデータセットを使って評価
verbose : Trueにすると各イテレーションでの評価指標などを表示
          イテレーションの軌跡を確認できる
"""
eval_set = [(X_test, y_test)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=True)

# %%
# 予測
from sklearn import metrics
y_pred = model.predict_proba(X_test)
metrics.log_loss(y_test, y_pred)
# splitしたテストデータを検証データとして使ってearly stoppingしたので、
# 検証データに対して過学習となっていることに注意
# 最終的なモデルとして相応しいかどうかは，kfoldCVで汎化性能を計測

# %%
# 特徴量の重要度
import matplotlib.pyplot as plt
# plt.barh(y_label, y_value)
plt.barh(X.columns, model.feature_importances_)

# %%




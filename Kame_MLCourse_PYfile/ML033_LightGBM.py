# %%
# LightGBMはscikit-learnの中に実装されていないのでインストール
%pip install lightgbm

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
n_estimators : ブースティングの回数(=学習する決定木の数) デフォルトは100
boosting_type : ブースティングアルゴリズムを以下から選択 デフォルトは 'gbdt'
                'gbdt' , 従来の Gradient Boosting Decision Tree
                'dart' , Dropouts meet Multiple Additive Regression Trees
                'goss' , Gradient-based One-Side Sampling
                'rf' , Random Forest.
learning_rate : shrinkageのη デフォルトは0.1
max_depth : 決定木の最高の深さ デフォルトではfull treeまで学習してしまうので
            何か値を入れておくとよい
random_state : 乱数のseed
"""
import lightgbm as lgb
model = lgb.LGBMClassifier(boosting_type='goss', max_depth=5, random_state=0)

# %%
# 学習
"""
eval_set : (X, y)のリスト 
           ブースティング時の各イテレーションごとにこのデータセットを使って評価
callbacks : 各イテレーションの際に実行するcallback関数のリスト(early stopping時に使用)
"""
eval_set = [(X_test, y_test)]
callbacks = []
# lgb.early_stopping() -> early_stopping
callbacks.append(lgb.early_stopping(stopping_rounds=10))
# lgb.log_evaluation() -> 各イテレーション時に評価指標を出力
callbacks.append(lgb.log_evaluation())
model.fit(X_train, y_train, eval_set=eval_set, callbacks=callbacks)

# %%
print(f'best_iterarion : {model.best_iteration_}')
print(f'best_score : {model.best_score_}')

# %%
# 予測
from sklearn import metrics
y_pred = model.predict_proba(X_test)
metrics.log_loss(y_test, y_pred)
# splitしたテストデータを検証データとして使ってearly stoppingしたので、
# 検証データに対して過学習となっていることに注意
# 最終的なモデルとして相応しいかどうかは，kfoldCVで汎化性能を計測

# %%
# 学習曲線(learning curve)
# 学習の各イテレーションで学習データや検証データの評価がどのように推移したかを示す．
# これを使ってパラメータをどのように変更すべきかを考えたりする．
# 例えばあまりにも収束が早い場合，学習率を下げた方が良かったり，
# 過学習気味であれば木をもう少し小さくするなどする.
# (過学習しているかどうかは学習データの学習曲線も必要)
lgb.plot_metric(model)

# %%
# 特徴量の重要度
lgb.plot_importance(model)

# %%




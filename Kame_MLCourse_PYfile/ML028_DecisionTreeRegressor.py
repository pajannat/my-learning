# %%
# データ準備
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# 決定木は特徴量ごとに分割していくので，特徴量間でスケールを合わせる必要なし
# 決定木では質的変数をそのまま扱うことができるが、
# scikit-learnは文字列の特徴量をサポートしていないのでダミー変数化が必要
# 機械学習をする場合、どんなモデルにも対応できるように、前処理でダミー変数化することが多い

df = sns.load_dataset('diamonds')
# one-hotエンコードを使ってダミー変数化
df = pd.get_dummies(df, drop_first=True)
X = df.loc[:, df.columns!='price']
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %%
# モデル学習
from sklearn import tree
# 引数をなにも指定しないと最後まで分割をし複雑なモデルになり過学習してしまう
# max_depth, min_samples_split, ccp_alpha などをよく使う
model = tree.DecisionTreeRegressor(max_depth=4)
model.fit(X_train, y_train)

# %%
# 予測
model.predict(X_test)

# %%
# 決定木の可視化
# 可視化その1 図で表示
import matplotlib.pyplot as plt
plt.figure(figsize=(40, 20))
# 戻り値は図の内容がテキストのリストで返ってくる
# 学習済みのtreeモデルを引き数に渡す
# feature_names 引数には特徴量の名前のリストを入れる
_ = tree.plot_tree(model, fontsize=10, feature_names=X.columns)

# %%
# 可視化その2 テキストで表示
print(tree.export_text(model, feature_names=list(X.columns)))

# %%
# 最初に分割する境界が最もRSSの合計を下げるようになっている
# -> 決定木の上に来ている特徴量は，目的変数を予測するのにより重要な特徴量

# 決定木の特徴
# 他のモデルよりも一般的には精度が落ちる
# 人が決定するロジックに近く解釈しやすい
# ロバスト性に欠ける。データの外れ値や、データの少しの変化で結果が大きく変わることがある
# 沢山の決定木を組み合わせることで精度の高いモデルを構築することができる

# %%




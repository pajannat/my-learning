# %%
# 線形回帰モデルを構築
from sklearn.linear_model import LinearRegression
# クラスのインスタンスを生成する
model = LinearRegression()
# 学習データを準備
import seaborn as sns
df = sns.load_dataset('diamonds')
df = df[(df[['x','y','z']] != 0).all(axis=1)]
X = df['carat'].values
y = df['price'].values

# %%
print(X.shape, y.shape)

# %%
# Xをscikit-learnで使用するためにMxNの形に成形
X = X.reshape(-1, 1)
print(X.shape)

# %%
# 学習する
model.fit(X, y)

# %%
# 学習済みのモデルのパラメータを確認
print(model.coef_, model.intercept_)

# %%
# 新たなデータで予測をする
import numpy as np
# carat=2のデータを用意
X_new = np.array(2).reshape(-1, 1)
# X_newに対して予測
model.predict(X_new)

# %%




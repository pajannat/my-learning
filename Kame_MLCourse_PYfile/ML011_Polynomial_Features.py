# %%
# データを準備
import seaborn as sns
df = sns.load_dataset('mpg')
df.dropna(inplace=True)
X = df['horsepower'].values.reshape(-1, 1)
y = df['mpg'].values

# %%
# scatter plotで描画
import matplotlib.pyplot as plt
sns.scatterplot(df['horsepower'], df['mpg'])

# %%
# PolynomialFeatures クラスを使って多項式特徴量に変換
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(2)
X_poly = poly.fit_transform(X)
X_poly

# %%
# LinearRegression() を使って線形回帰アルゴリズムでモデルを学習
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_poly, y)
# 1, x, x2に対する係数
model.coef_

# %%
import numpy as np
# X軸の値作成
x = np.arange(50, 230).reshape(-1, 1)
# .predictの前に同様にfit_transformする必要があることに注意
x_ = poly.fit_transform(x)
pred_ = model.predict(x_)
# 描画
sns.scatterplot(df['horsepower'], df['mpg'])
plt.plot(x, pred_, 'r')

# %%
# 特徴量に変換を行なって学習した場合，評価時にもデータに対して同様の変換が必要であることに注意
# 特徴量の変換など，学習時に前処理をした場合は
# テストデータに対しても同様に前処理をする必要がある

# %%




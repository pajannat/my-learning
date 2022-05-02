# %%
# データ取得
import numpy as np
import pandas as pd
import seaborn as sns

df = sns.load_dataset('diamonds')
df.head()

# %%
# 全体のデータの分布の確認
df.describe()

# %%
# 異常値の対応
df = df[(df[['x','y','z']] != 0).all(axis=1)]
len(df)

# %%
# caratとpriceのscatter plotを確認
import matplotlib.pyplot as plt
%matplotlib inline
carat = df['carat'].values
price = df['price'].values

plt.scatter(carat, price, alpha=0.1)
plt.xlabel('carat')
plt.ylabel('price')

# %%
# 正規方程式実装
X = np.vstack([np.ones(len(carat)), carat]).T
y = price
theta_best = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)

# %%
# 結果の可視化
x_axis = np.linspace(0, 3, 10)
y_pred = theta_best[0] + theta_best[1]*x_axis

plt.scatter(carat, price, alpha=0.1)
plt.plot(x_axis, y_pred, 'red')
plt.xlabel('carat')
plt.ylabel('price')

# %%




# %%
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
df = sns.load_dataset('diamonds')
df = df[(df[['x','y','z']] != 0).all(axis=1)]
X = df['depth'].values
y = df['price'].values
model = LinearRegression()
X = X.reshape(-1, 1)
model.fit(X, y)
print(model.coef_, model.intercept_)

# %%
X_ = np.append(np.ones((len(X),1)), X, axis=1)
theta = np.append(model.intercept_,model.coef_)
y_preds = model.predict(X)

# %%
RSS = np.sum((y-y_preds)**2)
RSE = np.sqrt(RSS/(len(X_)-len(X_[0])))
SE_sq = RSE**2 * np.linalg.inv(np.dot(X_.T,X_)).diagonal()

# %%
# t値を計算
t = theta/np.sqrt(SE_sq)
print(t)

# %%
# p値を計算
# p値 -> 帰無仮説 sita1=0 の下で、今回のsita1が得られる確率
from scipy import stats
p = [2*(1-stats.t.cdf(np.abs(t_val),(len(X_)-len(X_[0])))) for t_val in t]
print(p)

# %%
# depthとpriceのscatterplot
sns.scatterplot(x=df['depth'], y=df['price'], alpha=0.1)

# %%
# statsmodelsを使って一発でp値を確認する
import statsmodels.api as sma
X2 = sma.add_constant(X)
est = sma.OLS(y, X2)
est_trained = est.fit()
print(est_trained.summary())

# %%
# 複数の特徴量のp値を確認
import statsmodels.api as sma
X = df[['carat', 'depth', 'table', 'x', 'y', 'z']].values
X2 = sma.add_constant(X)
est = sma.OLS(y, X2)
est_trained = est.fit()
print(est_trained.summary())

# %%




# %%
# MSE(Mean Squared Error)
# mean_squared_error() に正解データ y_true と予測データ y_pred を入れる
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import seaborn as sns
model = LinearRegression()

# data prepare
df = sns.load_dataset('tips')
X = df['total_bill'].values.reshape(-1, 1)
y = df['tip'].values

# hold out
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

# train
model.fit(X_train, y_train)

# predict
y_pred = model.predict(X_test)

# evaluate
mean_squared_error(y_test, y_pred)

# %%
# RMSE(Root Mean Squared Error)
# mean_squared_error() で squared=False
mean_squared_error(y_test, y_pred, squared=False)

# %%
# MAE(Mean Absolute Error)
# シンプルに「誤差の平均」を確認できる
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_test, y_pred)

# %%
# R-Squared (決定係数 R^2)
# R^2 = 1 - RSS/TSS
# y のばらつきのうちXで説明できる割合
# 回帰の精度の指標として使える
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)

# %%
# adjuted R-Squared (自由度調整済み決定係数)
# 関係のないランダムな特徴量でも追加するとR^2が1に近づいてしまう
# これは特徴量を追加するだけでRSSが向上するため
# 回帰の精度を見る文脈では
# テストデータに対してR2を計算すればいいのでadjusted R-Squaredを計算する必要はない
# adjustedR^2 = 1 - {RSS/(m-n-1)}/{TSS/(m-1)}
# adjustedR^2 = 1 - {(1-R^2)*(m-1)}/(m-n-1)
r2 = r2_score(y_test, y_pred)
adj_r2 = 1-(1-r2)*(len(X_test)-1)/(len(X_test)-len(X_test[0])-1)
adj_r2

# %%
# 学習データに対してadjustedR^2を見る場合
import statsmodels.api as sma
X2 = sma.add_constant(X)
est = sma.OLS(y, X2)
est_trained = est.fit()
print(est_trained.summary())

# %%




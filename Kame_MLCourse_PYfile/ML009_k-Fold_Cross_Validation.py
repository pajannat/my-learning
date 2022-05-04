# %%
# k-Fold Cross ValidationをPythonで実装
import numpy as np
import seaborn as sns
df = sns.load_dataset('tips')
X = df['total_bill'].values.reshape(-1, 1)
y = df['tip'].values

# %%
from sklearn.linear_model import LinearRegression

# 5-Fold Cross Validation
from sklearn.model_selection import KFold
cv = KFold(n_splits=5, random_state=0, shuffle=True)
model = LinearRegression()
mse_list = []
for train_index, test_index in cv.split(X):
    # get train and test data
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    # fit model
    model.fit(X_train, y_train)
    # predict test data
    y_pred = model.predict(X_test)
    # loss
    mse = np.mean((y_pred - y_test)**2)
    mse_list.append(mse)
print(f"MSE(5FoldCV): {np.mean(mse_list)}")
print(f"std: {np.std(mse_list)}")

# %%
# cross_val_score 関数で一行でscoreを算出
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1)

# %%
np.mean(scores)

# %%




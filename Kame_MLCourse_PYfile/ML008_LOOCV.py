# %%
# PythonでLOOCVを実装
import seaborn as sns
import numpy as np
df = sns.load_dataset('tips')
X = df['total_bill'].values.reshape(-1, 1)
y = df['tip'].values

# %%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
model = LinearRegression()
mse_list = []
for train_index, test_index in loo.split(X):
#     print("train index:", train_index, "test index:", test_index)
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
print(f"MSE(LOOCV): {np.mean(mse_list)}")
print(f"std: {np.std(mse_list)}")

# %%
# LeaveOneOut メソッドの中身を見てみる
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
for train_index, test_index in loo.split(X):
    print("test index:", test_index, "train index:", train_index)

# %%




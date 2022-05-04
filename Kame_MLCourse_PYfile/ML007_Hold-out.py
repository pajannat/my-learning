# %%
import seaborn as sns
df = sns.load_dataset('diamonds')
df = df[(df[['x','y','z']] != 0).all(axis=1)]
X = df['carat'].values.reshape(-1, 1)
y = df['price'].values

# %%
# hold-outでデータを分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=0)

# %%
# 学習データのみでモデルをfit
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# %%




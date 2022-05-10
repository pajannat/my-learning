# %%
# データ準備
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

df = sns.load_dataset('titanic')
# 欠損値を除去
df = df.dropna()
X = df.loc[:, (df.columns!='survived') & (df.columns!='alive')]
# one-hotエンコードを使ってダミー変数化
X = pd.get_dummies(X, drop_first=True)
y = df['survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


# %%
from sklearn import tree
model = tree.DecisionTreeClassifier(random_state=0)
path = model.cost_complexity_pruning_path(X_train, y_train)
# Minimal Cost-Complexity Pruning をした際の係数を見てみる
eff_alphas, impurities = path.ccp_alphas, path.impurities
print(eff_alphas)
print(impurities)

# %%
# 各 effective alphaを用いてcost complexity pruningした際の結果を見てみる
# 各 effective alpha に対する model を models に格納
models = []
for eff_alpha in eff_alphas:
    model = tree.DecisionTreeClassifier(random_state=0, ccp_alpha=eff_alpha)
    model.fit(X_train, y_train)
    models.append(model)

# %%
# 各 model に対する精度を算出して格納、結果を出力
import matplotlib.pyplot as plt
train_scores = [model.score(X_train, y_train) for model in models]
test_scores = [model.score(X_test, y_test) for model in models]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.plot(eff_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
ax.plot(eff_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
ax.legend()
plt.show()

# %%
# alphaが小さいうち(つまりpruningなし)は木が複雑であるが故に過学習している
# 木をpruningしていくと，汎化性能が高くなっている
# 実際にはk-foldCVなどで汎化性能を測るのがよい

# %%




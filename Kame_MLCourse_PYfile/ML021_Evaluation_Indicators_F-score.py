# %%
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
 
# データロード
df = sns.load_dataset('iris')
# 学習データとテストデータ作成
X = df.loc[:, df.columns!='species']
y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
# モデル構築
model = LogisticRegression(penalty='none')
model.fit(X_train, y_train)
# 予測(ラベル)
y_pred = model.predict(X_test)
confusion_matrix(y_test, y_pred)

# %%
# F値 (F-score) RecallとPrecisionのトレードオフを考慮し調和平均を取った値
# Recallの逆数とPrecisionの逆数の平均、の逆数
from sklearn.metrics import f1_score
f1_score(y_test, y_pred, average='macro')

# %%
# データをdfに格納
import pandas as pd
df = pd.read_csv('heart.csv')
df.head()

# %%
# 異常値の削除
df = df[df['ca'] < 4] #drop the wrong ca values
df = df[df['thal'] > 0] # drop the wong thal value

# カラム名をもっとわかりやすい名前に変換
df = df.rename(columns = {'cp':'chest_pain_type', 'trestbps':'resting_blood_pressure', 'chol': 'cholesterol','fbs': 'fasting_blood_sugar', 
                       'restecg' : 'rest_electrocardiographic', 'thalach': 'max_heart_rate_achieved', 'exang': 'exercise_induced_angina',
                       'oldpeak': 'st_depression', 'slope': 'st_slope', 'ca':'num_major_vessels', 'thal': 'thalassemia'}, errors="raise")

# 質量変数の値がintegerになっているので，文字列にする(ついでにわかりやすい値を入れる
df['sex'][df['sex'] == 0] = 'female'
df['sex'][df['sex'] == 1] = 'male'

df['chest_pain_type'][df['chest_pain_type'] == 0] = 'typical angina'
df['chest_pain_type'][df['chest_pain_type'] == 1] = 'atypical angina'
df['chest_pain_type'][df['chest_pain_type'] == 2] = 'non-anginal pain'
df['chest_pain_type'][df['chest_pain_type'] == 3] = 'asymptomatic'

df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 0] = 'lower than 120mg/ml'
df['fasting_blood_sugar'][df['fasting_blood_sugar'] == 1] = 'greater than 120mg/ml'

df['rest_electrocardiographic'][df['rest_electrocardiographic'] == 0] = 'normal'
df['rest_electrocardiographic'][df['rest_electrocardiographic'] == 1] = 'ST-T wave abnormality'
df['rest_electrocardiographic'][df['rest_electrocardiographic'] == 2] = 'left ventricular hypertrophy'

df['exercise_induced_angina'][df['exercise_induced_angina'] == 0] = 'no'
df['exercise_induced_angina'][df['exercise_induced_angina'] == 1] = 'yes'

df['st_slope'][df['st_slope'] == 0] = 'upsloping'
df['st_slope'][df['st_slope'] == 1] = 'flat'
df['st_slope'][df['st_slope'] == 2] = 'downsloping'

df['thalassemia'][df['thalassemia'] == 1] = 'fixed defect'
df['thalassemia'][df['thalassemia'] == 2] = 'normal'
df['thalassemia'][df['thalassemia'] == 3] = 'reversable defect'

#質量変数をダミー変数にする
df = pd.get_dummies(df, drop_first=True)
df.head()
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# %%
# 学習データとテストデータ作成
X = df.loc[:, df.columns != 'target']
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# %%
# モデル構築
model = LogisticRegression(solver='liblinear')
model.fit(X_train, y_train)
# 予測(確率)
y_pred_proba = model.predict_proba(X_test)
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
%matplotlib inline

# %%
# 陽性の確率だけが必要なので[:, 1]をして陰性の確率を落とす
pos_prob = model.predict_proba(X_test)[:, 1]
precision, recall, thresholds =  precision_recall_curve(y_test, pos_prob)
# 描画
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.show()
import numpy as np
from scipy.stats import hmean

# %%
# F値計算
f1_scores = []
for p, r in zip(precision, recall):
    f1 = hmean([p, r])
    f1_scores.append(f1)
# Precision, Recall, F値を描画(vs 閾値)
plt.plot(thresholds, f1_scores[:-1], label='f1 score')
plt.plot(thresholds, recall[:-1], label='recall')
plt.plot(thresholds, precision[:-1], label='precision')
plt.xlabel('threshold')
plt.legend()
print(f'{np.argmax(f1_scores)}th threshold(={thresholds[np.argmax(f1_scores)]:.2f}) is highest f1 score ={np.max(f1_scores):.2f}')



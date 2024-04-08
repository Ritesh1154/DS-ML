import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

data = pd.read_csv('Admission_Predict.csv')

print(data.head())
print(data.isnull().sum())
data.columns = [c.replace(' ', '') for c in data.columns]

print(data.info())
data.loc[data['Admit']>0.80,'Admit'] = 1
data.loc[data['Admit']<=0.80,'Admit'] = 0
print((data == 0).sum())
print((data>=1).sum())

X = data.drop(['Admit','SerialNo.'],axis=1)
y = data['Admit']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model1 = DecisionTreeClassifier(max_depth= 3)
model1.fit(X_train,y_train)
y_pred = model1.predict(X_test)

y_train_pred = model1.predict(X_train)
y_test_pred = model1.predict(X_test)

print("Confusion matrix:\n")

print(metrics.confusion_matrix(y_test, y_pred))

print("1. Accuracy Score:", metrics.accuracy_score(y_test, y_pred))

print("2. Precision Score:",metrics.precision_score(y_test, y_pred))

print("3. Recall score : ",metrics.recall_score(y_test,y_pred))

print("4.F1_score : ",metrics.f1_score(y_test,y_pred))



tree.plot_tree(model1, filled=True, fontsize=10)
plt.show()


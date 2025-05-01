from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_auc_score, precision_recall_curve, precision_score, accuracy_score, \
    confusion_matrix,roc_curve
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from src.data.load_data import load_dataset
import numpy as np

mpl.rcParams['figure.dpi'] = 100
sns.set_style('white')


df = load_dataset()

y = df['default payment next month']
X = df['limit_bal']

X =  X.values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.33, random_state=42)

lr = LogisticRegression()
lr.fit(X_train,y_train)
print(lr.__dict__)

y_pred = lr.predict(X_train)

##### checking model fit

print('\n')
print(f'accuracy: {accuracy_score(y_train,y_pred):0.3f}%')
print('\n')
# print(f'Precision {precision_score(y_test,y_pred,zero_division=True)}')

print(f'AUC score {roc_auc_score(y_train,y_pred)}')

mat = confusion_matrix(y_train,y_pred)

###### confusion matrix

print(mat)
plt.figure()
sns.heatmap(mat,cbar=False,annot=True)
plt.xlabel('Y Predictions')
plt.ylabel('Y True')
plt.show()


######## Roc Auc

fpr,tpr, threhold = roc_curve(y_train,y_pred)

plt.figure()
plt.plot(fpr,tpr,'-k',label='Logistic Regression')
plt.plot([0,1],[0,1],'--r',label='Random')
plt.xlabel('False Positive Rate $FP/N$')
plt.ylabel('True Positive Rate $TP/P$')
plt.legend()
plt.title('ROC curve')
plt.show()
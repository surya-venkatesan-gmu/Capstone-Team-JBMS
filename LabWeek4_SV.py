#!/usr/bin/env python
# coding: utf-8

# In[68]:


#libraries bro 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sn
from sklearn.model_selection import cross_val_predict, KFold


# In[62]:


# Tasks 1 - 4

# read file in with pandas
df = pd.read_csv("C://Users//15713//Desktop//CO2_data.csv")
print(df.head()) # to see our column names so we can select 

# setting x
x = df[['Weight', 'Volume']]

# setting y
y = df['CO2 Above 100']

# linear regression
regr = linear_model.LinearRegression()
regr.fit(x, y)
print(regr.coef_) # [0.00025895 0.0003905 ]
# interpretation:
# A 1 unit increase in weight increases prediction by 0.00025895
# A 1 unit increase in volume increases prediction by 0.0003905
# thus we can conclude volume plays a slightly larger role than weight in the prediction of CO2 Above 100


# In[88]:


# Task 5 --> cross fold validation --> train/test split
# full disclusre, learned about the KFold method from here https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
our_folds = KFold(n_splits=5, shuffle=True, random_state=42)
regr_cv_strat = linear_model.LinearRegression()
predicted_y = cross_val_predict(regr_cv_strat, x, y, cv=our_folds)

# Task 6 --> continous into binary
predicted_binary = (predicted_y >= 0.5).astype(int)
print("Actuals vs. Prediction")
print(f'Actuals: {y.to_numpy()}')
print(f'Predict: {predicted_binary}')
acc = accuracy_score(y, predicted_binary)
print(f'Accuracy: {acc}')


# In[90]:


# Task 7 confusion and classificaiton
cm = confusion_matrix(y, predicted_binary, labels=[0, 1])
tn, fp, fn, tp = cm.ravel().tolist()
print(cm)
print(f'True Neg: {tn}')
print(f'False Positive: {fp}')
print(f'False Neg: {fn}')
print(f'True Positive: {tp}')
# [[14  5]
#  [ 7 10]]
# True Neg: 14
# False Positive: 5
# False Neg: 7
# True Positive: 10

# classification report
print(classification_report(y, predicted_binary))
#                 precision    recall  f1-score   support

#            0       0.67      0.74      0.70        19
#            1       0.67      0.59      0.62        17

#     accuracy                           0.67        36
#    macro avg       0.67      0.66      0.66        36
# weighted avg       0.67      0.67      0.66        36


# In[114]:


# Final Task - plotting the confusion matrix
# full disclosure, took help from this website to do this: https://towardsdatascience.com/heatmap-for-confusion-matrix-in-python-20a9fc689665/
cm_visual = pd.DataFrame(cm, index=[0, 1], columns=[0, 1])
plt.figure(figsize=(4,4))
cfm_plot = sn.heatmap(cm_visual, annot=True, fmt='d', cmap="Greens")
plt.xlabel("predictions")
plt.ylabel("actual")
plt.title("confusion matrix")
plt.show()


# In[ ]:





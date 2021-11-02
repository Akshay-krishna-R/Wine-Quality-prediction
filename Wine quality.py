#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os 
import warnings
warnings.filterwarnings('ignore')


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[4]:


data_set = pd.read_csv(r"Final Project 1.csv")


# In[5]:


data_set.head()


# In[6]:


data_set.info()


# In[7]:


data_set.isnull().sum()


# In[8]:


data_set.describe()


# In[9]:


data_set["quality"].unique()


# In[10]:


data_set["quality"].nunique()


# In[12]:


correlation_dataset = data_set.corr()


# In[13]:


plt.figure(figsize=(10,8))
sns.heatmap(correlation_dataset, cbar = True, square=True, fmt='.1g', annot= True,  annot_kws = {'size':8}, cmap='Greens')


# In[14]:


x = data_set.drop(columns=['quality'], axis=1)
y = data_set['quality']


# In[15]:


y.value_counts()


# In[16]:


x.head()


# In[17]:


y.head()


# In[18]:


X_train, X_test, Y_train, Y_test  =  train_test_split(x, y,stratify = y, test_size=0.10, random_state=0)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[19]:


Y_train.value_counts()


# In[21]:


Y_test.value_counts()


# In[22]:


sclaer = StandardScaler()
sclaer.fit(X_train)
X_train = sclaer.transform(X_train)
X_test = sclaer.transform(X_test)


# In[24]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200, random_state=0,max_depth=20)
rfc.fit(X_train, Y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test, y_pred)
print("Random forest model test score : ", score*100)


# In[25]:


y_pred_train = rfc.predict(X_train)

from sklearn.metrics import accuracy_score
score = accuracy_score(Y_train, y_pred_train)
print("Random forest model train score : ", score*100)


# In[26]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[29]:


Y_train.value_counts()


# In[27]:


print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[30]:


data_set["quality"].value_counts()


# In[ ]:





# In[ ]:





# In[31]:


count_classes = pd.value_counts(data_set["quality"],sort = True)


# In[34]:


print(count_classes)


# In[38]:


count_classes.plot(kind="bar",rot=0)
plt.title("Class distribution")
plt.xlabel("Classes")
plt.ylabel("Frequency")


# In[43]:


from imblearn.over_sampling import SMOTE
from collections import Counter


# In[50]:


counter = Counter(Y_train)
counter


# In[51]:


smt = SMOTE()
X_train_sm,Y_train_sm = smt.fit_resample(X_train,Y_train)


# In[52]:


counter = Counter(Y_train_sm)
counter


# In[54]:


count_classes = Y_train_sm.value_counts()
count_classes.plot(kind="bar",rot=0)
plt.title("Class distribution")
plt.xlabel("Classes")
plt.ylabel("Frequency")


# In[55]:


X_train_sm.shape,Y_train_sm.shape


# In[56]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=200, random_state=0,max_depth=20)
rfc.fit(X_train_sm, Y_train_sm)
y_pred = rfc.predict(X_test)

from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test, y_pred)
print("Random forest model test score : ", score*100)


# In[58]:


y_pred_train = rfc.predict(X_train_sm)

from sklearn.metrics import accuracy_score
score = accuracy_score(Y_train_sm, y_pred_train)
print("Random forest model train score : ", score*100)


# In[59]:


print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[69]:


correlation_dataset = data_set.corr()


# In[70]:


plt.figure(figsize=(10,8))
sns.heatmap(correlation_dataset, cbar = True, square=True, fmt='.1g', annot= True,  annot_kws = {'size':8}, cmap='Greens')


# In[82]:


def highly_correlation_function(dataset,threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix)):
        if abs(corr_matrix.iloc[len(corr_matrix)-1,i]) >=threshold:
            col_name = corr_matrix.columns[i]
            col_corr.add(col_name)
        
    return col_corr


# In[72]:


dataset_highly_corr = highly_correlation_function(data_set,0.2)
len(dataset_highly_corr)


# In[73]:


x = data_set[dataset_highly_corr]
y = data_set['quality']


# In[74]:


X_train, X_test, Y_train, Y_test  =  train_test_split(x, y, test_size=0.10, random_state=2,stratify = y)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[75]:


sclaer = StandardScaler()
sclaer.fit(X_train)
X_train = sclaer.transform(X_train)
X_test = sclaer.transform(X_test)


# In[76]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=4, random_state=0)
rfc.fit(X_train, Y_train)
y_pred = rfc.predict(X_test)


# In[77]:


from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test, y_pred)
print("Random forest model test score : ", score*100)


# In[78]:


y_pred_train = rfc.predict(X_train)
score = accuracy_score(Y_train, y_pred_train)
print("Random forest model train score : ", score*100)


# In[79]:


print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[84]:


dataset_highly_corr = highly_correlation_function(data_set,0.2)
len(dataset_highly_corr)


# In[85]:


dataset_highly_corr


# In[86]:


x = data_set[dataset_highly_corr]
y = data_set['quality']


# In[87]:


X_train, X_test, Y_train, Y_test  =  train_test_split(x, y, test_size=0.10, random_state=2,stratify = y)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[88]:


sclaer = StandardScaler()
sclaer.fit(X_train)
X_train = sclaer.transform(X_train)
X_test = sclaer.transform(X_test)


# In[89]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=4, random_state=0)
rfc.fit(X_train, Y_train)
y_pred = rfc.predict(X_test)


# In[90]:


from sklearn.metrics import accuracy_score
score = accuracy_score(Y_test, y_pred)
print("Random forest model test score : ", score*100)


# In[91]:


y_pred_train = rfc.predict(X_train)
score = accuracy_score(Y_train, y_pred_train)
print("Random forest model train score : ", score*100)


# In[92]:


print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[93]:


smt = SMOTE()
X_train_sm,Y_train_sm = smt.fit_resample(X_train,Y_train)


# In[94]:


count_classes = Y_train_sm.value_counts()
count_classes.plot(kind="bar",rot=0)
plt.title("Class distribution")
plt.xlabel("Classes")
plt.ylabel("Frequency")


# In[95]:


Y_train_sm.value_counts()


# In[96]:


sclaer = StandardScaler()
sclaer.fit(X_train)
X_train = sclaer.transform(X_train)
X_test = sclaer.transform(X_test)


# In[98]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=4, random_state=0)
rfc.fit(X_train_sm, Y_train_sm)
y_pred_test = rfc.predict(X_test)
y_pred_train = rfc.predict(X_train_sm)



score = accuracy_score(Y_test, y_pred_test)
print("Random forest model test score : ", score*100)


score = accuracy_score(Y_train_sm, y_pred_train)
print("Random forest model train score : ", score*100)


# In[99]:


print(classification_report(Y_test, y_pred_test))
print(confusion_matrix(Y_test, y_pred))


# In[ ]:





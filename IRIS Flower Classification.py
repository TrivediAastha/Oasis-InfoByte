#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


data = pd.read_csv('Downloads/Iris.csv')
data


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data.drop_duplicates()


# # EDA
# 

# In[8]:


data.hist(figsize=(20,10),bins=50)


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


sns.pairplot(data)


# In[11]:


sns.countplot(data['SepalLengthCm'])


# In[12]:


sns.countplot(data['SepalWidthCm'])


# In[13]:


sns.countplot(data['PetalLengthCm'])


# In[14]:


sns.countplot(data['PetalWidthCm'])


# In[15]:


h = data.columns
h


# In[16]:


sns.boxplot(x=data["SepalLengthCm"])


# In[17]:


sns.boxplot(x=data["SepalWidthCm"])


# In[18]:


sns.boxplot(x=data["PetalLengthCm"])


# In[19]:


sns.boxplot(x=data["PetalWidthCm"])


# In[20]:


data.corr()


# In[21]:


sns.heatmap(data.corr(),annot=True)


# # Model Evaluation

# In[22]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[23]:


X = data.drop('Species',axis=1)
Y = data['Species']


# In[24]:


X_train,X_test,Y_train,Y_test = train_test_split(X ,Y, test_size=0.3, random_state=42)


# # LogisticRegression

# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


model = LogisticRegression()
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[27]:


print("Accuracy:",accuracy_score(Y_test,pred))
print("*"*50)
print(confusion_matrix(Y_test,pred))
print("*"*50)
print(classification_report(Y_test,pred))


# # DecisionTree

# In[28]:


from sklearn.tree import DecisionTreeClassifier


# In[29]:


model = DecisionTreeClassifier()
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[30]:


print("Accuracy:",accuracy_score(Y_test,pred))
print("*"*50)
print(confusion_matrix(Y_test,pred))
print("*"*50)
print(classification_report(Y_test,pred))


# # RandomForest

# In[31]:


from sklearn.ensemble import RandomForestClassifier


# In[32]:


model = RandomForestClassifier()
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[33]:


print("Accuracy:",accuracy_score(Y_test,pred))
print("*"*50)
print(confusion_matrix(Y_test,pred))
print("*"*50)
print(classification_report(Y_test,pred))


# # SVC

# In[34]:


from sklearn.svm import SVC


# In[35]:


model = SVC()
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[36]:


print("Accuracy:",accuracy_score(Y_test,pred))
print("*"*50)
print(confusion_matrix(Y_test,pred))
print("*"*50)
print(classification_report(Y_test,pred))


# # KNN

# In[37]:


from sklearn.neighbors import KNeighborsClassifier


# In[38]:


model = KNeighborsClassifier()
model.fit(X_train,Y_train)
pred = model.predict(X_test)


# In[39]:


print("Accuracy:",accuracy_score(Y_test,pred))
print("*"*50)
print(confusion_matrix(Y_test,pred))
print("*"*50)
print(classification_report(Y_test,pred))


# # Accuracy Plotting

# In[40]:


Classifier = ('log_re','Dec_tr','Ran_Fo','Svc','Knn')
Accuracy = (1.0,1.0,1.0,1.0,1.0)


# In[41]:


plt.bar(Classifier,Accuracy)


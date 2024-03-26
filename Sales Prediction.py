#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# # LOAD DATA

# In[2]:


data = pd.read_csv('Downloads/Advertising.csv')


# In[3]:


data


# In[4]:


data.head()


# # DATA PREPROCESSING

# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


data.dropna()


# In[9]:


data.drop_duplicates()


# # EXPLORATORY DATA ANALYSIS

# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[11]:


sns.pairplot(data)


# In[12]:


sns.histplot(data)


# In[13]:


data.corr()


# In[14]:


sns.heatmap(data.corr(),annot=True)


# # TRAINING AND TESTING DATA

# In[15]:


X = data.drop('Sales', axis=1)
y = data['Sales']


# In[16]:


X


# In[17]:


y


# In[18]:


from sklearn.model_selection import train_test_split


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# # LINEAR REGRESSION

# In[20]:


from sklearn.linear_model import LinearRegression


# In[21]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[22]:


pred = model.predict(X_test)


# In[23]:


pred


# In[24]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[26]:


print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('R-squared:', r2_score(y_test, pred))


# In[28]:


model.score(X_train,y_train)


# In[30]:


model.score(X_test,y_test)


# In[32]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result


# # RANDOM FOREST REGRESSOR

# In[33]:


from sklearn.ensemble import RandomForestRegressor


# In[35]:


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# In[36]:


y_pred = model.predict(X_test)


# In[37]:


print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))


# In[38]:


model.score(X_train,y_train)


# In[39]:


model.score(X_test,y_test)


# In[40]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result


# # SVR

# In[41]:


from sklearn.svm import SVR


# In[42]:


model = SVR(kernel='linear')
model.fit(X_train, y_train)


# In[43]:


y_pred = model.predict(X_test)


# In[44]:


print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('R-squared:', r2_score(y_test, pred))


# In[45]:


model.score(X_train,y_train)


# In[46]:


model.score(X_test,y_test)


# In[47]:


result2 = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result2


# # DECISION TREE REGRESSOR

# In[48]:


from sklearn.tree import DecisionTreeRegressor


# In[49]:


model = DecisionTreeRegressor(ccp_alpha=0.0)
model.fit(X_train, y_train)


# In[50]:


y_pred = model.predict(X_test)


# In[51]:


print('Mean Squared Error:', mean_squared_error(y_test, pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, pred))
print('R-squared:', r2_score(y_test, pred))


# In[52]:


model.score(X_train,y_train)


# In[53]:


model.score(X_test,y_test)


# In[54]:


result3 = pd.DataFrame({'Actual': y_test, 'Predicted': pred})
result3


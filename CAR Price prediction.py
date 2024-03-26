#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# # LOAD DATA

# In[2]:


data = pd.read_csv('Desktop/cars.csv')
data


# In[3]:


data.head()


# # DATA PREPROCESSING

# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data.drop_duplicates()


# In[8]:


data.dropna()


# # EXPLORATORY DATA ANALYSIS(EDA)

# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[10]:


sns.pairplot(data)


# In[11]:


sns.histplot(data)


# In[12]:


data.corr()


# In[13]:


sns.heatmap(data.corr(),annot=True)


# In[14]:


pd.get_dummies(data['Fuel_Type'],drop_first=False)


# In[15]:


data


# # SPLITTING TRAINING AND TESTING DATA

# In[16]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[17]:


X=data.drop(['Selling_Price'],axis=1)
y=data['Selling_Price']


# In[18]:


X


# In[19]:


y


# In[20]:


X_train,X_test,Y_train,Y_test = train_test_split(X,y,test_size=0.3,random_state=42)


# In[21]:


X_train


# In[22]:


Y_train


# In[23]:


X = pd.get_dummies(X)


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[25]:


from sklearn.preprocessing import StandardScaler


# In[26]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# # LINEAR REGRESSION

# In[86]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[87]:


model = LinearRegression()
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)


# In[88]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# In[89]:


print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))


# In[90]:


model.score(X_train_scaled,y_train)


# In[91]:


model.score(X_test_scaled,y_test)


# In[92]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
result


# # RANDOM FOREST REGRESSOR

# In[93]:


from sklearn.ensemble import RandomForestRegressor


# In[94]:


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)


# In[95]:


y_pred = model.predict(X_test_scaled)


# In[96]:


print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))


# In[97]:


model.score(X_train_scaled,y_train)


# In[98]:


model.score(X_test_scaled,y_test)


# In[99]:


result1 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
result1


# # SVR

# In[100]:


from sklearn.svm import SVR


# In[101]:


model = SVR(kernel='linear')
model.fit(X_train_scaled, y_train)


# In[102]:


y_pred = model.predict(X_test_scaled)


# In[103]:


print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))


# In[104]:


model.score(X_train_scaled,y_train)


# In[105]:


model.score(X_test_scaled,y_test)


# In[106]:


result2 = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
result2


# # DECISION TREE REGRESSOR

# In[107]:


from sklearn.tree import DecisionTreeRegressor


# In[108]:


model = DecisionTreeRegressor(ccp_alpha=0.0)
model.fit(X_train_scaled, y_train)


# In[109]:


y_pred = model.predict(X_test_scaled)


# In[110]:


print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
print('Mean Absolute Error:', mean_absolute_error(y_test, y_pred))
print('R-squared:', r2_score(y_test, y_pred))


# In[111]:


model.score(X_train_scaled,y_train)


# In[112]:


model.score(X_test_scaled,y_test)


# In[113]:


result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
result


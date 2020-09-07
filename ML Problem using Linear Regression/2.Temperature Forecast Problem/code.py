#!/usr/bin/env python
# coding: utf-8

# In[35]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[36]:


data = pd.read_csv('Bias_correction_ucl.csv')


# In[37]:


data.head()


# In[38]:


data.info()


# In[39]:


sns.heatmap(data.isnull(), yticklabels=False, cbar= False, cmap= 'viridis')


# In[40]:


data.dropna()


# In[41]:


data.info()


# In[42]:


sns.heatmap(data.isnull(), yticklabels=False, cbar= False, cmap= 'viridis')


# In[43]:


data.dropna(axis=0, inplace=True)


# In[44]:


data.info()


# In[45]:


data.head()


# In[46]:


data['Date'].unique()


# In[51]:


data['Date']= pd.to_datetime(data['Date'])
data['day of week']=data['Date'].apply(lambda x: x.dayofweek)
data['month'] = data['Date'].apply(lambda x: x.month)
data['year'] =data['Date'].apply(lambda x: x.year)


# In[52]:


data.head()


# In[53]:


data.drop(['Date'], axis=1, inplace=True)


# In[54]:


data.head()


# In[55]:


data['day of week'].unique()


# In[ ]:





# In[56]:


data.head()


# In[58]:


X = data.drop(['Next_Tmax','Next_Tmin'], axis=1)


# In[61]:


y= data[['Next_Tmax','Next_Tmin']]


# In[62]:


X.head()


# In[63]:


y.head()


# In[64]:


from sklearn.model_selection import train_test_split


# In[65]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[66]:


from sklearn.linear_model import LinearRegression


# In[67]:


lr = LinearRegression()


# In[82]:


lr.fit(X_train, y_train)


# In[83]:


lr.intercept_


# In[ ]:





# In[ ]:





# In[86]:


predd=lr.predict(X_test)


# In[87]:


from sklearn import metrics


# In[88]:


print('MAE: ',metrics.mean_absolute_error(y_test, predd))
print('MSE: ',metrics.mean_squared_error(y_test, predd))
print('RMSE: ',np.sqrt(metrics.mean_squared_error(y_test, predd)))
print('R^2: ',metrics.explained_variance_score(y_test, predd))


# In[90]:


plt.figure(figsize=(14,8))
sns.set_palette('GnBu_r')
sns.set_style('darkgrid')
plt.scatter(y_test, predd)


# In[98]:


plt.figure(figsize=(14,8))
sns.distplot((y_test-predd),bins=50);


# In[ ]:





# In[ ]:





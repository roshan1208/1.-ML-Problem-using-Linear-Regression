#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('Ecommerce Customers')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.columns


# In[7]:


sns.pairplot(df)


# In[8]:


sns.distplot(df['Yearly Amount Spent'])


# In[9]:


df.head()


# In[10]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[11]:


sns.jointplot(x='Time on App', y='Yearly Amount Spent', data = df, color='red')


# In[ ]:





# In[12]:


sns.jointplot(x='Time on Website', y='Yearly Amount Spent', data = df)


# In[13]:


sns.jointplot(x='Avg. Session Length', y='Yearly Amount Spent', data = df)


# In[14]:


sns.jointplot(x='Length of Membership', y='Yearly Amount Spent', data = df)


# In[15]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=df)


# In[16]:


sns.heatmap(df.corr())


# In[17]:


X = df[['Avg. Session Length', 'Time on App',
       'Time on Website', 'Length of Membership']]


# In[18]:


y = df['Yearly Amount Spent']


# In[19]:


X


# In[20]:


y


# In[21]:


from sklearn.model_selection import train_test_split


# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=101)


# In[23]:


X_train


# In[24]:


X_test


# In[25]:


y_train


# In[26]:


y_test


# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


lm= LinearRegression()


# In[29]:


lm.fit(X_train, y_train)


# In[30]:


print(lm.intercept_)


# In[31]:


df_coeff = pd.DataFrame(lm.coef_, X.columns,['Coefficient'])


# In[32]:


df_coeff


# In[33]:


predictions = lm.predict(X_test)


# In[34]:


predictions


# In[84]:


y_test


# In[35]:


plt.scatter(predictions, y_test)


# In[36]:


sns.distplot(y_test-predictions, bins=30)


# In[37]:


y_test-predictions


# In[38]:


from sklearn import metrics


# In[39]:


print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# In[40]:


metrics.explained_variance_score(y_test,predictions)


# In[ ]:





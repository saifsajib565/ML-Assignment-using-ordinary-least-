#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


dict= {"Weight" : [2,4,5,3,6,5,7], 
"Price" : [35,60,20,50,50,55,60]
}
df = pd.DataFrame(dict)
df


# In[27]:


df1=pd.read_csv("C:/Users/saifu/Downloads/Aiquest Data science an machine learning materials/Class 04 Regression linear/Wight_Price.csv")
df1


# In[4]:


df.info()


# In[5]:


df[['Weight']]


# In[6]:


plt.scatter(df.Weight, df.Price, marker='*', color='red')	
plt.xlabel ("WEIGHT")
plt.ylabel ("PRICE")


# In[7]:


df.Weight.mean()


# In[8]:


df.Price.mean()


# In[9]:


df.head()


# In[10]:


x = df.drop( 'Price', axis=1)
x.head()


# In[11]:


y = df.drop( 'Weight', axis=1)
y.head()


# In[12]:


from sklearn.linear_model import LinearRegression


# In[13]:


reg=LinearRegression()


# In[14]:


reg.fit(x,y)


# In[15]:


reg.coef_


# In[16]:


reg.intercept_


# In[17]:


reg.predict([[6]])


# In[18]:


df['residuals']= df[['Weight']] - reg.predict(x)
df.head()


# In[19]:


df['predict'] = reg.predict(x)
df.head()


# In[20]:


plt.scatter(df.Weight, df.predict, marker='*', color='red')	
plt.xlabel ("WEIGHT")
plt.ylabel ("PRICE")


# In[21]:


plt.scatter(df.Weight, df.predict, marker='o', color='green')	
plt.scatter(df.Weight, df.Price, marker='*', color='red')
plt.xlabel ("WEIGHT")
plt.ylabel ("PRICE")


# In[22]:


plt.plot(df.Weight, df.predict, marker='o', color='green')	
plt.scatter(df.Weight, df.Price, marker='*', color='red')
plt.xlabel ("WEIGHT")
plt.ylabel ("PRICE")


# In[23]:


from sklearn.metrics import mean_squared_error, mean_absolute_error


# In[24]:


mse = mean_squared_error (df['Price'], df['predict'])
mse


# In[25]:


mean_squared_error (df['Price'], df['predict'])


# In[26]:


mean_absolute_error (df['Price'], df['predict'])


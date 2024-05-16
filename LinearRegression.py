#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import pandas as pd
from sklearn import linear_model
from word2number  import w2n


# In[28]:


df = pd.read_csv('hiring.csv')


# In[29]:


df


# In[30]:


import math
test_score =math.floor(df['test_score(out of 10)'].median())
test_score


# In[31]:


df['test_score(out of 10)']= df['test_score(out of 10)'].fillna(test_score)
df


# In[32]:


df.experience = df.experience.fillna('zero')
df


# In[35]:


df.experience = df.experience.apply(w2n.word_to_num)
df


# In[38]:


reg = linear_model.LinearRegression()
reg.fit(df[['experience','test_score(out of 10)','interview_score(out of 10)']],df['salary($)'])


# In[39]:


reg.predict([[0,10,10]])


# In[47]:


reg.predict([[10,6,8]])


# In[ ]:





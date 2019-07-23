
# coding: utf-8

# In[1]:


import pandas as pd


# In[4]:


# 예측 유저 목록
result = pd.read_csv('result.csv', names=['user_id'])


# In[6]:


result.to_csv("recommend.txt", header=False, index=False)
print("[Success!] recommend.txt")


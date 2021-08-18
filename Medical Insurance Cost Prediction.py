#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")


# ## Reading the data

# In[2]:


insurance_dataset = pd.read_csv('downloads\insurance.csv')


# In[3]:


insurance_dataset.head()


# In[4]:


insurance_dataset.describe()


# In[5]:


insurance_dataset.shape


# In[6]:


insurance_dataset.info()


# In[7]:


insurance_dataset.isnull().sum()


# In[8]:


plt.bar(insurance_dataset["age"],insurance_dataset["charges"],color='brown')
plt.show()


# In[53]:


plt.hist(insurance_dataset["bmi"],color='green')
plt.show()


# In[9]:


plt.scatter(insurance_dataset['age'],insurance_dataset['charges'])
plt.show()


# In[10]:


plt.scatter(insurance_dataset['age'],insurance_dataset['bmi'],color='black')
plt.show()


# In[11]:


sns.set()
plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['age'],color='red')
plt.title('Age Distribution')
plt.show()


# In[12]:


plt.figure(figsize=(6,6))
sns.countplot(x='sex', data=insurance_dataset)
plt.title('Sex Distribution')
plt.show()


# In[14]:


insurance_dataset['sex'].value_counts()


# In[13]:


plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['bmi'],color='cyan')
plt.title('BMI Distribution')
plt.show()


# In[14]:


plt.figure(figsize=(6,6))
sns.countplot(x='children', data=insurance_dataset)
plt.title('Children')
plt.show()


# In[15]:


insurance_dataset['children'].value_counts()


# In[16]:


plt.figure(figsize=(6,6))
sns.countplot(x='smoker', data=insurance_dataset)
plt.title('smoker')
plt.show()


# In[17]:


insurance_dataset['smoker'].value_counts()


# In[18]:


plt.figure(figsize=(6,6))
sns.countplot(x='region', data=insurance_dataset)
plt.title('region')
plt.show()


# In[19]:


insurance_dataset['region'].value_counts()


# In[20]:


plt.figure(figsize=(6,6))
sns.distplot(insurance_dataset['charges'],color="green")
plt.title('Charges Distribution')
plt.show()


# ## Data Pre-Processing 

# In[21]:


insurance_dataset.replace({'sex':{'male':0,'female':1}}, inplace=True)


# In[22]:


insurance_dataset.replace({'smoker':{'yes':0,'no':1}}, inplace=True)


# In[23]:


insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}}, inplace=True)


# In[24]:


X = insurance_dataset.drop(columns='charges', axis=1)
Y = insurance_dataset['charges']


# In[25]:


X


# In[26]:


Y


# ## Splitting the data into Training data & Testing Data 

# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# In[28]:


print(X.shape, X_train.shape, X_test.shape)


# ## Model Training

# In[29]:


regressor = LinearRegression()


# In[30]:


regressor.fit(X_train, Y_train)


# In[31]:


training_data_prediction =regressor.predict(X_train)


# In[32]:


r2_train = metrics.r2_score(Y_train, training_data_prediction)
print('R squared mean value : ', r2_train)


# In[33]:


test_data_prediction =regressor.predict(X_test)


# In[34]:


r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R squared mean value : ', r2_test)


# ## Predicting with test values

# In[35]:


input_data = (31,1,25.74,0,1,0)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
prediction = regressor.predict(input_data_reshaped)
print(prediction)

print('The insurance cost is USD $',prediction[0])


# In[ ]:





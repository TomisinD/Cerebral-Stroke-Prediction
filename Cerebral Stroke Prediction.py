#!/usr/bin/env python
# coding: utf-8

# ### Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import warnings
warnings.filterwarnings ('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Importing Data Set

# In[2]:


Cerebral_stroke =pd.read_csv("Stroke.csv")


# ### Viewing the First Five Rows

# In[3]:


Cerebral_stroke.head()


# ### Viewing the Last Five Rows

# In[4]:


Cerebral_stroke.tail()


# ### Exploring Data Information

# In[5]:


Cerebral_stroke.describe()


# In[6]:


Cerebral_stroke.info()


# In[7]:


Cerebral_stroke.shape


# In[8]:


Cerebral_stroke.columns


# In[9]:


Cerebral_stroke.isnull().sum()


# In[10]:


print('percentage of missing:\n',(Cerebral_stroke.isnull().sum()/len(Cerebral_stroke)*100))


# In[11]:


Cerebral_stroke.duplicated().sum()


# In[12]:


Cerebral_stroke.smoking_status.value_counts()


# In[13]:


Cerebral_stroke.stroke.value_counts()


# In[14]:


Cerebral_stroke.gender.value_counts()


# In[15]:


Cerebral_stroke.work_type.value_counts()


# In[16]:


Cerebral_stroke.heart_disease.value_counts()


# In[17]:


print(Cerebral_stroke[['age']].max())
print(Cerebral_stroke[['age',]].min())


# In[18]:


Cerebral_stroke.isnull().sum()


# In[19]:


Cerebral_stroke.dropna(inplace=True)


# In[20]:


Cerebral_stroke.isnull().sum()


# ### Data Visualization

# In[21]:


Cerebral_stroke.hist(figsize=(20,10))


# In[22]:


plt.figure(figsize=(20, 10))
sns.countplot(x='gender', hue='stroke', data=Cerebral_stroke, palette=sns.color_palette("rocket"))
plt.title('Stroke by Gender')
plt.legend(title='stroke', loc='upper right')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt


# In[23]:


plt.figure(figsize=(20, 10))
sns.countplot(x='work_type', hue='stroke', data=Cerebral_stroke, palette=sns.color_palette("rocket"))
plt.title('Stroke by Work Type')
plt.legend(title='Stroke', loc='upper right')
plt.xlabel('Work Type')
plt.ylabel('Count')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt


# In[24]:


plt.figure(figsize=(20, 10))
sns.countplot(x='ever_married', hue='stroke', data=Cerebral_stroke, palette=sns.color_palette("rocket"))
plt.title('Stroke by Marital Status')
plt.legend(title='Stroke', loc='upper right')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()


# In[25]:


plt.figure(figsize=(20, 10))
sns.countplot(x='hypertension', hue='stroke', data=Cerebral_stroke, palette=sns.color_palette("rocket"))
plt.title('Stroke by Hypertension Status')
plt.legend(title='Stroke', loc='upper right')
plt.xlabel('Hypertension Status Status')
plt.ylabel('Count')
plt.show()


# In[65]:


plt.figure(figsize=(20, 10))
sns.countplot(x='age', hue='stroke', data=Cerebral_stroke, palette=sns.color_palette("rocket"))
plt.title('Stroke by Age group')
plt.legend(title='Stroke', loc='upper right')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.show()


# ### Data Imputation

# In[26]:


Cerebral_stroke.head()


# In[27]:


Encoded =list(Cerebral_stroke)
print(Encoded)


# In[28]:


le =LabelEncoder()
for i in Encoded:
    Cerebral_stroke[i] =le.fit_transform(Cerebral_stroke[i])  


# In[29]:


Cerebral_stroke.head()


# In[30]:


plt.figure(figsize=(20,10))
corr=Cerebral_stroke.corr()
sns.heatmap(corr,cmap="Accent", annot =True)
corr


# ### Normalization

# In[31]:


from sklearn.preprocessing import normalize


# In[32]:


Data=Cerebral_stroke


# In[33]:


Normalized_data= normalize(Data)


# In[34]:


Normalized_data


# In[35]:


Normalized_data =pd.DataFrame(Normalized_data, columns=Data.columns)


# In[36]:


Normalized_data.head()


# ### Model Building

# #### Specifying X and Y Labels

# In[37]:


y=Normalized_data[['stroke']].values
x=Normalized_data.drop(['id'], axis=1).values


# #### Training the Model

# In[38]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)


# In[39]:


x_train.shape


# ### Linear Regression

# In[40]:


Linear_regression =LinearRegression()
Linear_regression.fit(x_train,y_train)


# #### Making Predictions

# In[41]:


y_predict=Linear_regression.predict(x_test)
print(y_predict)
print(y_test)


# #### Printing Model Performance

# In[42]:


print('coefficient:',Linear_regression.coef_)


# In[43]:


print('intercept:',Linear_regression.intercept_)


# In[44]:


print('Mean Square Error:%8f' %mean_squared_error(y_test,Linear_regression.predict(x_test)))


# In[45]:


r1=metrics.r2_score(y_test,y_predict)


# ### Random Forest Regression

# #### Spliting the Data

# In[46]:


y=Normalized_data[['stroke']].values
x=Normalized_data.drop(['id'], axis=1).values


# ### Training the Models

# In[47]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20, random_state=0)


# In[48]:


RandomForest =RandomForestRegressor()
RandomForest.fit(x_train,y_train)


# #### Predictions

# In[49]:


y_predict2=RandomForest.predict(x_test)
print(y_predict)
print(y_test)


# #### Performance Evaluation of Model

# In[50]:


r2=metrics.r2_score(y_test,y_predict2)


# ### models Performance Comparison

# In[51]:


DataF=pd.DataFrame({'Models':['Linear_regression','RandomForest'], 'R2':[r1,r2]})


# In[52]:


DataF


# In[ ]:





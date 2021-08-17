#!/usr/bin/env python
# coding: utf-8

# # GRIP: The Sparks Foundation #AUG21
# ### Data Science and Business Analytics Intern 
# ### Name: Srihari Reddy Kata
# ### Task 1: Prediction using Machine Learning Algorithm
# 

# In this task we have to predict the percantage of student based on the number of hours he/she studied.This task have two variables where the feature is no.of hours studied and target value is percantage of score.This can be solved using simple linear regression model.

# In[1]:


#importing libraries required
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# #### Reading data from URL

# In[2]:


url='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'
data=pd.read_csv(url)
data.head()


# In[7]:


data.shape


# #### Exploring data

# In[3]:


data.describe()


# In[9]:


data.info()


# In[4]:


#Plotting Dataset values using plot
data.plot(kind='scatter',x='Hours',y='Scores')
plt.show()


# ##### From the above Plot we can clearly say that it follows Linear Regression Model as the dependent variable is increasing linearly with exploratory variable

# In[ ]:


hours=data['Hours']
scores=data['Scores']
sns.distplot(hours)


# data.corr(method='spearman')

# In[7]:


sns.distplot(scores)


# ### Linear Regression

# In[13]:


X=data.iloc[:,:1].values
y=data.iloc[:,1].values


# In[14]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)


# ### Training the Model

# In[ ]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)


# In[16]:


m= reg.coef_
c=reg.intercept_
line=m*X+c
plt.scatter(X,y)
plt.plot(X,line)
plt.show()


# In[17]:


y_pred = reg.predict(X_test)


# In[18]:


actual_predicted = pd.DataFrame({'Target':y_test,'Predicted':y_pred})
actual_predicted


# In[19]:


sns.set_style('whitegrid')
sns.distplot(np.array(y_test-y_pred))
plt.show()


# What would be the predicted score if a student scores for 9.25 hours/day?

# In[20]:


h=9.25
s=reg.predict([[h]])
print("If a student studies for {} hours per day he/she will score {}% in exam".format(h,s))


# ## Model Evaluation

# In[21]:


from sklearn import metrics
from sklearn.metrics import r2_score
print('Mean Absolute Error:',metrics.mean_absolute_error(y_test,y_pred))
print('R2 score:',r2_score(y_test,y_pred))


# In[ ]:





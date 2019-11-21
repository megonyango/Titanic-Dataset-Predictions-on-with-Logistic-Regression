#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


train = pd.read_csv('titanic_train.csv')
test  = pd.read_csv('titanic_test.csv')


# In[3]:


train.head()


# In[4]:


test.head()


# # EDA 

# In[6]:


# Using heat map to check for missing values on the data sets
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[7]:


sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[9]:


#number of those who survived in the train set

sns.set_style('whitegrid')
sns.countplot(x='Survived', data=train)


# In[12]:


#rows contained in the data test set

test.columns

#the test set does not contain thw Survival data since that is what we wish to predict


# In[13]:


#visualization of those who survived categorized on age

sns.countplot(x='Survived', data=train, hue='Sex')

#more female passangers seemed to survive compared to the male counterparts


# In[14]:


sns.countplot(x='Survived', data=train, hue='Pclass')

#more passangers seemed to survive were on first class 


# # Now on the ages 

# In[16]:


#train data
sns.distplot(train['Age'].dropna(),kde=False,bins=30)


# In[17]:


sns.distplot(test['Age'].dropna(),kde=False,bins=30)


# In[ ]:





# In[20]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age', data=train)


# In[21]:



plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass',y='Age', data=test)


# # Cleaning our Data 

# In[30]:


#a function that puts in our missing values on the age

def impute_age_train(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


# In[32]:


train['Age'] = train[['Age','Pclass']].apply(impute_age_train, axis=1)


# In[33]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[34]:



def impute_age_test(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 42

        elif Pclass == 2:
            return 27

        else:
            return 26

    else:
        return Age


# In[35]:


test['Age'] = test[['Age','Pclass']].apply(impute_age_test, axis=1)


# In[36]:


sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[37]:


#test drop cabin
train.drop('Cabin',axis=1,inplace=True)
test.drop('Cabin',axis=1,inplace=True)


# In[40]:


print(train.columns)
print(test.columns)


# In[41]:


#check for missing data using heatmap
sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[42]:


sns.heatmap(test.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[43]:


train.head()


# In[46]:


#convert sex data to numerical values

sex_train=pd.get_dummies(train['Sex'],drop_first=True)
sex_test=pd.get_dummies(test['Sex'],drop_first=True)

print(sex_train)
print('----------------------')
print(sex_test)


# In[47]:


#convert  embark numerical values

embark_train=pd.get_dummies(train['Embarked'],drop_first=True)
embark_test=pd.get_dummies(test['Embarked'],drop_first=True)


print(embark_train)
print('----------------------')
print(embark_test)


# In[50]:


#concatanate the two colums for both train and test

train= pd.concat([train,sex_train,embark_train],axis=1)
test= pd.concat([test,sex_test,embark_test],axis=1)


# In[52]:


test.head()


# In[54]:


#drop all the columns that will not be resourcefull in our simple model
train.drop(['Name', 'Sex','Ticket', 'Embarked','PassengerId'],axis=1, inplace=True)


# In[56]:


train.head()


# In[57]:


test.drop(['Name', 'Sex','Ticket', 'Embarked','PassengerId'],axis=1, inplace=True)


# In[58]:


test.head()


# # The Machine Learning Model

# In[77]:


X=train.drop('Survived',axis=1)
y=train['Survived']


# In[ ]:


#trainning the models


# In[72]:


from sklearn.model_selection import train_test_split


# In[74]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[78]:


from sklearn.linear_model import LogisticRegression


# In[81]:


logmodel=LogisticRegression()


# In[82]:


logmodel.fit(X_train,y_train) 


# In[94]:


predictions=logmodel.predict(X_test)


# In[95]:


test.dropna(inplace=True)


# In[96]:


predictions_for_test=logmodel.predict(test)


# In[97]:


from sklearn.metrics import classification_report


# In[103]:


predictions_for_test


# # Out Put

# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-plot')


# In[2]:


#import libraries
from sklearn import preprocessing
import pandas as pd

#import different ML classifiers
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

#import ML evaluation metrics
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import metrics, model_selection
from sklearn.model_selection import cross_val_score
from  sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import LeavePOut


#import scikitplot to plot confusion matrix
import scikitplot as skplt


# In[4]:


df = pd.read_csv("Updated-Dataset.csv")


# In[5]:


features = df.drop('gender', axis=1)
results = df['gender']


# In[6]:


labels = preprocessing.LabelEncoder()
results_encoded = labels.fit_transform(results)
results_encoded


# In[7]:


features['beard'] = labels.fit_transform(features['beard'])
features['hair_length'] = labels.fit_transform(features['hair_length'])
features['scarf'] = labels.fit_transform(features['scarf'])
features['eye_color'] = labels.fit_transform(features['eye_color'])
print(features['hair_length'])


# In[8]:


Features_train, features_test, Results_train, results_test = train_test_split(features, results, test_size = 0.058, random_state = 2)


# In[9]:


model = GaussianNB()


# In[10]:


model.fit(Features_train,Results_train)


# In[11]:


prediction = model.predict(features_test)


# In[12]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[20]:


model_cl_rep = metrics.classification_report(results_test, prediction)
print(model_cl_rep)


# In[22]:


print('End')


# In[ ]:





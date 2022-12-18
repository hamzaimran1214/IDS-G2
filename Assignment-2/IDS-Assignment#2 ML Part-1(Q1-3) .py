
#FA19-BCS-131
#Syed Hamza imran
#IDS-Assignment no 2




#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-plot')


# In[74]:


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


# In[3]:


df = pd.read_csv("gender-prediction.csv")


# In[5]:
# List All the Instances Classifid as Male

df[df['gender'].str.count('^[m].*')>0]


# In[6]:


#Instances Classified as Female
df[df['gender'].str.count('^[f].*')>0]


# In[7]:
#Spreating Depandent and Independent Varaibales


features = df.drop('gender', axis=1)
results = df['gender']


# In[10]:
# Encoding Independent Varaibles


labels = preprocessing.LabelEncoder()
results_encoded = labels.fit_transform(results)
results_encoded


# In[11]:
# Encoding Dependent Varaibales

features['beard'] = labels.fit_transform(features['beard'])
features['hair_length'] = labels.fit_transform(features['hair_length'])
features['scarf'] = labels.fit_transform(features['scarf'])
features['eye_color'] = labels.fit_transform(features['eye_color'])
print(features['hair_length'])


# In[12]:


#make train/test split
Features_train, features_test, Results_train, results_test = train_test_split(features, results, test_size = 0.33, random_state = 2)


# In[13]:
# UsinG Random Forest


model = RandomForestClassifier()


# In[15]:


model.fit(Features_train,Results_train)


# In[19]:


prediction = model.predict(features_test)


# In[20]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[21]:
#Using MLP Classifier


model=MLPClassifier()


# In[22]:


model.fit(Features_train,Results_train)


# In[23]:


prediction = model.predict(features_test)


# In[24]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[26]:
# Using SVC Model


model = SVC()


# In[27]:


model.fit(Features_train,Results_train)


# In[28]:


prediction = model.predict(features_test)


# In[29]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[30]:


#now Changing Split Ratio
Features_train, features_test, Results_train, results_test = train_test_split(features, results, test_size = 0.20, random_state = 2)


# In[31]:


model = RandomForestClassifier()


# In[32]:


model.fit(Features_train,Results_train)


# In[33]:


prediction = model.predict(features_test)


# In[34]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[35]:


model=MLPClassifier()


# In[36]:


model.fit(Features_train,Results_train)


# In[37]:


prediction = model.predict(features_test)


# In[38]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[39]:


model = SVC()


# In[40]:


model.fit(Features_train,Results_train)


# In[41]:


prediction = model.predict(features_test)


# In[42]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[53]:
# <---------Question  no 2 ------------>
# <---------Droping The Important Tow Variables--------->


new_Features = df.drop(['scarf','beard','gender'], axis=1)


# In[54]:


new_Features['hair_length'] = labels.fit_transform(new_Features['hair_length'])
new_Features['eye_color'] = labels.fit_transform(new_Features['eye_color'])


# In[55]:


#now Changing Split Ratio
NEW_Features_train, new_Features_test, Results_train, results_test = train_test_split(new_Features, results, test_size = 0.20, random_state = 2)


# In[56]:


model = RandomForestClassifier()


# In[57]:


model.fit(NEW_Features_train,Results_train)


# In[59]:


prediction = model.predict(new_Features_test)


# In[60]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[61]:


model=MLPClassifier()


# In[62]:


model.fit(NEW_Features_train,Results_train)


# In[63]:


prediction = model.predict(new_Features_test)


# In[64]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[65]:


model=SVC()


# In[66]:


model.fit(NEW_Features_train,Results_train)


# In[67]:


prediction = model.predict(new_Features_test)


# In[68]:


model_acc = accuracy_score(results_test, prediction)*100
print(model_acc)


# In[71]:


#Q3
#Applying Decision Tree Classifier 
mc = ShuffleSplit(n_splits=5,test_size=0.33,random_state=2)
model =  DecisionTreeClassifier()


# In[73]:


dec_tree = DecisionTreeClassifier()
mean_score = cross_val_score(dec_tree, features, results_encoded, scoring="f1", cv = mc).mean()
mean_score


# In[76]:


# Leave P Out Cross Validation
dec_tree = DecisionTreeClassifier()
lpo=LeavePOut(p=2)
lpo.get_n_splits(features)
score=cross_val_score(dec_tree, features, results_encoded, cv=lpo).mean()
score


# In[ ]:





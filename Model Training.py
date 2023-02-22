#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from random import randint
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# ### Loading Data From Train File & Combining data with Test to increase amount of Data For Training

# In[2]:


train = pd.read_csv(r'D:\Loan Prediction System\Loan_Data\train.csv')
train.Loan_Status = train.Loan_Status.map({'Y':1,'N':0}) # This line will be used for mapping the data.
train.head()


# In[3]:


train.shape # Data rows & columns


# In[4]:


train.columns


# In[5]:


train.describe() # Data Described From the train.csv file


# In[6]:


train.isnull().sum()  # Calculating null values from each column 


# In[7]:


# Adding test & train data & creating loan status for model trainig as it will be output variable.
Loan_status=train.Loan_Status
train.drop('Loan_Status',axis=1,inplace=True)
test = pd.read_csv(r'D:\Loan Prediction System\Loan_Data\test.csv')
Loan_ID = test.Loan_ID
data = train.append(test)
data.head()


# In[8]:


data.shape #rows and columns after combining data 


# In[9]:


data.describe()


# In[10]:


data.isnull().sum()  # we got more null values now when data combined


# ### Starting to Map Data for Model Training

# In[11]:


data.Dependents.dtypes


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[13]:


# Label Encoding for Gender. Encoding to prepare data for model training
data.Gender=data.Gender.map({'Male':1,'Female':0})
data.Gender.value_counts()


# In[14]:


data.head() # Looking at head to verify the encoding on gender


# In[15]:


# correlation increased after encoding based on Gender
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[16]:


# Label Encoding for Marital Status
data.Married=data.Married.map({'Yes':1,'No':0})
data.Married.value_counts()


# In[17]:


# label Encoding for Dependents
data.Dependents=data.Dependents.map({'0':0,'1':1,'2':2,'3+':3})
data.Dependents.value_counts()


# In[18]:


# Let's see correlations for Dependents & Marital Status
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[19]:


# Labelling 0 & 1 for Education Status
data.Education=data.Education.map({'Graduate':1,'Not Graduate':0})
data.Education.value_counts()


# In[20]:


## Labelling 0 & 1 for Employment status
data.Self_Employed=data.Self_Employed.map({'Yes':1,'No':0})
data.Self_Employed.value_counts()


# In[21]:


# Labelling 0 & 1 for Property area
data.Property_Area=data.Property_Area.map({'Urban':2,'Rural':0,'Semiurban':1})
data.Property_Area.value_counts()


# In[22]:


# Final Correlation Matrix 
corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[23]:


# Finally All Data Converted into Numeric Values
data.head()


# In[24]:


# Looking at the number of records in the credit_history column
data.Credit_History.size


# In[25]:


# Fillin the empty columns, removing null values
data.Credit_History.fillna(np.random.randint(0,2),inplace=True)
data.isnull().sum()


# In[26]:


# Filling null values with random 1's and 0's in Married Column
data.Married.fillna(np.random.randint(0,1),inplace=True)
data.isnull().sum()


# In[27]:


## Filling null values with median for loanAmount as it is value for which median can be better choice
data.LoanAmount.fillna(data.LoanAmount.median(),inplace=True)
data.isnull().sum()


# In[28]:


# Filling null values with mean
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(),inplace=True)
data.isnull().sum()


# In[29]:


# Filling Gender Null Values with 1's and 0's
data.Gender.fillna(np.random.randint(0,1),inplace=True)
data.Gender.value_counts()


# In[30]:


data.isnull().sum()


# In[31]:


# Filling Dependents null values with median
data.Dependents.fillna(data.Dependents.median(),inplace=True)
data.isnull().sum()


# In[32]:


data.isnull().sum()


# In[33]:


corrmat=data.corr()
f,ax=plt.subplots(figsize=(9,9))
sns.heatmap(corrmat,vmax=.8,square=True)


# In[34]:


# lastly with self_Employed Null columns filled our data is prepared for training model.
data.Self_Employed.fillna(np.random.randint(0,1),inplace=True)
data.isnull().sum()


# In[35]:


data.head()


# In[36]:


# Dropping Loan ID from data, it's not useful
data.drop('Loan_ID',inplace=True,axis=1)
data.isnull().sum()


# In[37]:


data.head()


# ### Here Onwards Data Split & Model Training begins 

# In[38]:


train_X=data.iloc[:614,] ## all the data in X (Train set) which is till 614 records
train_y=Loan_status  ## Loan status will be our Y


# In[39]:


# DATA SPLIT FROM TEST TO TRAIN
train_X,test_X,train_y,test_y=train_test_split(train_X,train_y,random_state=0)
train_X.head()


# In[40]:


test_X.head()


# In[41]:


models=[]
models.append(("Logistic Regression",LogisticRegression()))
models.append(("Decision Tree",DecisionTreeClassifier()))
models.append(("Linear Discriminant Analysis",LinearDiscriminantAnalysis()))
models.append(("Random Forest",RandomForestClassifier()))
models.append(("Support Vector Classifier",SVC()))
models.append(("K- Neirest Neighbour",KNeighborsClassifier()))
models.append(("Naive Bayes",GaussianNB()))


# In[42]:


scoring='accuracy'
result=[]
names=[]


# In[43]:


# Training Data with Different Models to find better Accurracy
for name,model in models:
    kfold=KFold(n_splits=10,random_state=0)
    cv_result=cross_val_score(model,train_X,train_y,cv=kfold,scoring=scoring)
    result.append(cv_result)
    names.append(name)
    print(model)
    print("%s %f" % (name,cv_result.mean()))


# In[44]:


# Finding the model accurracy with score, it gives about 81% accuracy
LR=LogisticRegression()
train_X.head()
LR.fit(train_X,train_y)
pred=LR.predict(test_X)
print("Model Accuracy:- ",accuracy_score(test_y,pred))
print(confusion_matrix(test_y,pred))
print(classification_report(test_y,pred))


# In[45]:


# Checking results of prediction from model
print(pred)


# In[46]:


# seperating test data into the X_test as from row 614 & onwards was data that was added
X_test=data.iloc[614:,]


# In[47]:


X_test.head()


# In[48]:


# finding predicion of model by testing it 
prediction = LR.predict(X_test)
print(prediction)


# In[49]:


t = LR.predict([[0.0,	0.0,	0.0,	1,	0.0,	1811,	1666.0,	54.0,	360.0,	1.0,	2]])
print(t)


# In[50]:


# now you can save it to a file
with open('model_pkl', 'wb') as f:
    pickle.dump(LR, f)


# In[ ]:





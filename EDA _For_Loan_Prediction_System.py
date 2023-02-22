#!/usr/bin/env python
# coding: utf-8

# ## Exploratory Data Analysis

# In[39]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# In[5]:


# Reading Data and Looking at top 5 records
train = pd.read_csv(r"D:\Loan Prediction System\Loan_Data\train.csv")
train.head()


# In[6]:


# Looking at unique values from proerty_Area column
train['Property_Area'].value_counts()


# In[17]:


# Looking at unique values from Education column
train['Education'].value_counts()


# In[21]:


# Looking at unique values from creadit history
train['Credit_History'].value_counts()


# In[7]:


# Describing Data Set
train.describe()


# In[8]:


# Looking for null values in Dataset
train.isnull().sum()


# In[10]:


# Number of rows in Train datafram is 614
train


# In[9]:


# Seperating Columns Loan Status & loan ID from rest of data and combining test and train csv files
Loan_status = train.Loan_Status
train.drop('Loan_Status', inplace=True, axis=1)
test = pd.read_csv(r"D:\Loan Prediction System\Loan_Data\test.csv")
Loan_ID = test.Loan_ID
data = train.append(test)
data


# In[11]:


# Again looking for null values in New Data Frame
data.isnull().sum()


# In[12]:


# Mapping Gender Column this also gives us idea of number of male and female in dataset
data.Gender = data.Gender.map({'Male': 1, 'Female': 0})
print("Mapping Gender Column \n", data.Gender.value_counts())


# In[13]:


# Mapping Married Column 
data.Married = data.Married.map({'Yes': 1, 'No': 0})
print("Mapping Married Column \n", data.Married.value_counts())


# In[14]:


# Mapping Dependents Column
data.Dependents = data.Dependents.map({'0': 0, '1': 1, '2': 2, '3+': 3})
print("Mapping Dependents Column \n", data.Dependents.value_counts())


# In[15]:


# Mapping Education Column
data.Education = data.Education.map({'Graduate': 1, 'Not Graduate': 0})
print("Mapping Education Column \n", data.Education.value_counts())


# In[16]:


# Mapping self_employement Column
data.Self_Employed = data.Self_Employed.map({'Yes': 1, 'No': 0})
print("Mapping Self Employed Column \n", data.Self_Employed.value_counts())


# In[18]:


# Mapping Property_Area Column
data.Property_Area = data.Property_Area.map({'Urban': 2, 'Rural': 0, 'Semiurban': 1})
print("Mapping Property Area Column \n", data.Property_Area.value_counts())


# ## Filling Missing Values

# In[20]:


# Filling NUll values credit_hitory
data.Credit_History.fillna(np.random.randint(0, 2), inplace=True)
data.isnull().sum()


# In[22]:


looking_credit_history = data['Credit_History'].tolist()
print(looking_credit_history)


# In[23]:


# Filling NUll values Married
data.Married.fillna(np.random.randint(0, 2), inplace=True)
data.isnull().sum()


# In[24]:


# Filling Null Values Loan_Amount with median because median gives more sense for the column
data.LoanAmount.fillna(data.LoanAmount.median(), inplace=True)
data.isnull().sum()


# In[25]:


# Filling Null Values Loan_Amount_terms with mean because mean gives more sense for the column
data.Loan_Amount_Term.fillna(data.Loan_Amount_Term.mean(), inplace=True)
data.isnull().sum()


# In[26]:


# Filling Null Values Gender 
data.Gender.fillna(np.random.randint(0, 2), inplace=True)
data.isnull().sum()


# In[27]:


# Filling null values with median as there are 4 different values in the column 
data.Dependents.fillna(data.Dependents.median(), inplace=True)
data.isnull().sum()


# In[28]:


# Filling Null values in self-Employed column
data.Self_Employed.fillna(np.random.randint(0, 2), inplace=True)
data.isnull().sum()


# ## Since the dataset is small, Null were Filled instead of removing them 

# In[29]:


data


# In[40]:


# Splitting Data into X & Y For Training Model
train_X = data.iloc[:614, ]  # all the data in X (Train set)
train_y = Loan_status  # Loan status will be our
train_X, test_X, train_y, test_y = train_test_split(train_X, train_y, random_state=0)
train_X


# ## Data Visualization 

# In[47]:


# Adding Loan Status back into the train dataframe
train['Loan_status'] = Loan_status.values
train


# In[51]:


# Gender based Loan Amount Approved
graph = sns.FacetGrid(train,hue="Gender",height=9)
graph.map(plt.scatter, "Loan_status", "LoanAmount").add_legend()
plt.show()


# In[60]:


plt.figure(figsize=(10, 7))
x = train['LoanAmount']
plt.hist(x, bins= 30, color='pink')
plt.title("Loan Taken by Customers")
plt.xlabel("Loan figures")
plt.ylabel("Count")
plt.show()


# In[54]:


# Income of People With Reference to Area
graph1 = sns.FacetGrid(train, hue="Property_Area", height=9)
graph1.map(plt.scatter, "ApplicantIncome", "CoapplicantIncome").add_legend()
plt.show()


# In[65]:


# Graduates SelfEmployed
df_temp= train[train['Education'] == 'Graduate']
df_temp['Self_Employed'].hist()
plt.title("Number of Graduates who are SelfEmployed")
plt.xlabel("Yes For selfEmployed")
plt.ylabel("Count")
plt.show()


# In[67]:


# Co-Relation Matrix
corrmat = train.corr()
f, ax = plt.subplots(figsize=(9, 9))
sns.heatmap(corrmat, vmax=.8, square=True,annot = True)


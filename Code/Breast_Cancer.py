#!/usr/bin/env python
# coding: utf-8

# <div class="alert alert-block alert-success" style="text-align: center; border-radius: 5px;">
#     <b>Breast Cancer - Exploratory Data Analysis.</b>
# </div>
# 

# <div style="background-color: #cce5ff; color: #004085; padding: 5px; border: 1px solid #b8daff; border-radius: 5px; text-align: center;">
#     <b>Import Libabries.</b>
# </div>
# 

# In[1]:


# Import Different Libabries

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.graph_objs as go
import plotly.offline as py


# <div style="background-color: #cce5ff; color: #004085; padding: 5px; border: 1px solid #b8daff; border-radius: 5px; text-align: center;">
#     <b>Load Datasets</b>
# </div>
# 

# In[2]:


# Import dataset as csv file

cancer = pd.read_csv("Breast_Cancer.csv")


# <div style="background-color: #cce5ff; color: #004085; padding: 5px; border: 1px solid #b8daff; border-radius: 5px; text-align: center;">
#     <b>Data Preparation</b>
# </div>
# 

# ### Data Description

# In[3]:


# Preview the dataset

cancer


# ### Rename Columns

# In[4]:


# Rename Columns

cancer.columns = ['Age','Race','Marital_status','T_stage','N_stage','6th_stage','Differentiate','Grade','A_stage','Tumor_size','Estrogen_status','Progesterone_status','Regional_node_examined','Reginol_node_positive','Survival_months','Status']


# In[5]:


# Recheck the dataset

cancer.head()


# In[6]:


duplicates = cancer[cancer.duplicated()]

cancer_cleaned = cancer.drop_duplicates()

cancer.to_csv('Breast_Cancer.csv', index=False)

cancer


# ### Check out Missing Values

# In[7]:


# removing the dupliacte value:
cancer.drop_duplicates(inplace=True)
cancer.reset_index(inplace=True , drop=True)


# In[8]:


# Check missing values

print("Missing values per column:\n")
cancer.isnull().sum()


# In[9]:


# Check dimension of the dataset

print(f'Shape of the dataset')
print(f'Number of Attributes: {cancer.shape[1]}')
print(f'Number of Observations: {cancer.shape[0]}')


# In[10]:


# Check out data types

cancer.info()


# In[11]:


# Check the numeric and non-numeric attributes

total_attributes = cancer.shape[1]

numeric_attributes = cancer.select_dtypes(include=['number']).columns
num_numeric_attributes = len(numeric_attributes)


non_numeric_attributes = cancer.select_dtypes(exclude=['number']).columns
num_non_numeric_attributes = len(non_numeric_attributes)

print(f"Total number of attributes: {total_attributes}")
print(f"Total number of numeric attributes: {num_numeric_attributes}")
print(f"Total number of non-numeric attributes: {num_non_numeric_attributes}")


# <div style="background-color: #cce5ff; color: #004085; padding: 5px; border: 1px solid #b8daff; border-radius: 5px; text-align: center;">
#     <b>Exploratory Data Analysis</b>
# </div>

# ### Target Variable - Status

# In[12]:


# Make a pie chart of 'Status' values

def condition_ratio(data):
    
    results = data['Status'].value_counts()
    
    labels = results.index.tolist()
    values = results.values.tolist()

    colors = ['Green', 'Red']  

    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker={'colors': colors, 'line': {'color': 'Black', 'width': 2}}
    )])

    fig_pie.update_layout(title_text='Alive/Dead Status Distribution')
    fig_pie.show()
    
    
condition_ratio(cancer);


# 84.7% patients are alive while 15.3% patients are dead.

# ### Feature Variable - Marital

# In[13]:


# Make a pie chart of 'Marital Status' values in the data

def marriage_ratio(data):
    
    results = data['Marital_status'].value_counts()

    labels = results.index.tolist()
    values = results.values.tolist()

    colors = ['Red', 'Blue', 'Yellow', 'Brown', 'Green']  
    
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker={'colors': colors, 'line': {'color': 'Black', 'width': 2}}
    )])

    fig_pie.update_layout(title_text='Marital Status Distribution')
    fig_pie.show()
    
marriage_ratio(cancer)


# 65.7%, 15.3%, 12.1%, 5.84% and 1.12% patients are married, single, divorced, widowed,
# and separated respectively.

# In[14]:


# Make a pie chart of 'Race' values in the data

def race(data):
    
    results = data['Race'].value_counts()

    labels = results.index.tolist()
    values = results.values.tolist()

    
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker={'line': {'color': 'Black', 'width': 2}}
    )])

    fig_pie.update_layout(title_text='Race(White/Black) Distribution')
    fig_pie.show()
    
race(cancer)


# 84.8% of the patients are white and 7.23% patients are black.

# In[15]:


# Create a distribution plot for tumor size

plt.figure(figsize=(10,6))
sns.histplot(cancer['Tumor_size'], kde=True, bins=30, color='blue')
plt.title('Distribution of Tumor Size')
plt.xlabel('Tumor Size (in mm or cm, depending on the unit)')
plt.ylabel('Frequency')
plt.show()


# The average tumor size among the patients is 25 mm.

# In[16]:


# Make a bar chart and trendline of 'Survival Months' values

sns.histplot(cancer['Survival_months'], bins=30, kde=True)
plt.title('Distribution of Survival Months')
plt.show()


# The majority of the patients survive between 50 to 100 months after diagnosis.

# In[17]:


# Create a bar chart showing the correlation between marital status and alive/deceased status

def marital_status_proportion_status(data):

    data['Marital_status'].groupby(data['Status']).value_counts(normalize=True).rename('proportion').reset_index().pipe((sns.barplot, 'data'), x='Marital_status', y='proportion', hue='Status', palette='Dark2');
    plt.title('Proportion of Status for Martial Status')
    plt.xlabel('Martial Status')
    plt.show()
    
    
marital_status_proportion_status(cancer)


# Married patients have higher survival rates compared to others.

# In[18]:


# Create a plot between T Stage and Status

T = pd.crosstab(cancer['T_stage'], cancer['Status'])

T.plot.bar()


# A large number of patients in the T1 and T2 stages survive

# In[19]:


# Create a plot between N Stage and Status
N = pd.crosstab(cancer['N_stage'], cancer['Status'])

N.plot.bar()


# A large number of patients in the N1 stage survive.

# In[20]:


# Create a plot between Race and Status
race = pd.crosstab(cancer['Race'],cancer['Status'])

race.plot.bar()


# In[21]:


# Create a boxplot showing the correlation between tumor size and race

plt.figure(figsize=(10, 6))
sns.boxplot(x='Race', y='Tumor_size', data=cancer)
plt.title("Tumor Size by Race")
plt.show()


# In[22]:


# Create scatter plots for cancer data analysis with survival months

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
fig.suptitle("Cancer Data Analysis")

# Age vs. Survival Months
sns.scatterplot(x='Age', y='Survival_months', hue='Status', data=cancer, ax=axes[0, 0])
axes[0, 0].set_title("Age vs. Survival Months")

# Tumor Size vs. Survival Months
sns.scatterplot(x='Tumor_size', y='Survival_months', hue='Status', data=cancer, ax=axes[0, 1])
axes[0, 1].set_title("Tumor Size vs. Survival Months")

# Regional Node Examined vs. Survival Months
sns.scatterplot(x='Regional_node_examined', y='Survival_months', hue='Status', data=cancer, ax=axes[1, 0])
axes[1, 0].set_title("Regional Node Examined vs. Survival Months")

# Regional Node Positive vs. Survival Months
sns.scatterplot(x='Reginol_node_positive', y='Survival_months', hue='Status', data=cancer, ax=axes[1, 1])
axes[1, 1].set_title("Regional Node Positive vs. Survival Months")

plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()


# In[23]:


# Create pair plot for cancer data analysis

sns.pairplot(cancer)


# In[24]:


# Create a Correlation Heatmap

correlation_matrix = cancer.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='viridis')
plt.title("Correlation Heatmap")
plt.show()


# Correlation between reginol node positive and regional node examined are highest.

# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ## Portfolio Exercise: Starbucks
# <br>
# 
# <img src="https://opj.ca/wp-content/uploads/2018/02/New-Starbucks-Logo-1200x969.jpg" width="200" height="200">
# <br>
# <br>
#  
# #### Background Information
# 
# The dataset you will be provided in this portfolio exercise was originally used as a take-home assignment provided by Starbucks for their job candidates. The data for this exercise consists of about 120,000 data points split in a 2:1 ratio among training and test files. In the experiment simulated by the data, an advertising promotion was tested to see if it would bring more customers to purchase a specific product priced at $10. Since it costs the company 0.15 to send out each promotion, it would be best to limit that promotion only to those that are most receptive to the promotion. Each data point includes one column indicating whether or not an individual was sent a promotion for the product, and one column indicating whether or not that individual eventually purchased that product. Each individual also has seven additional features associated with them, which are provided abstractly as V1-V7.
# 
# #### Optimization Strategy
# 
# Your task is to use the training data to understand what patterns in V1-V7 to indicate that a promotion should be provided to a user. Specifically, your goal is to maximize the following metrics:
# 
# * **Incremental Response Rate (IRR)** 
# 
# IRR depicts how many more customers purchased the product with the promotion, as compared to if they didn't receive the promotion. Mathematically, it's the ratio of the number of purchasers in the promotion group to the total number of customers in the purchasers group (_treatment_) minus the ratio of the number of purchasers in the non-promotional group to the total number of customers in the non-promotional group (_control_).
# 
# $$ IRR = \frac{purch_{treat}}{cust_{treat}} - \frac{purch_{ctrl}}{cust_{ctrl}} $$
# 
# 
# * **Net Incremental Revenue (NIR)**
# 
# NIR depicts how much is made (or lost) by sending out the promotion. Mathematically, this is 10 times the total number of purchasers that received the promotion minus 0.15 times the number of promotions sent out, minus 10 times the number of purchasers who were not given the promotion.
# 
# $$ NIR = (10\cdot purch_{treat} - 0.15 \cdot cust_{treat}) - 10 \cdot purch_{ctrl}$$
# 
# For a full description of what Starbucks provides to candidates see the [instructions available here](https://drive.google.com/open?id=18klca9Sef1Rs6q8DW4l7o349r8B70qXM).
# 
# Below you can find the training data provided.  Explore the data and different optimization strategies.
# 
# #### How To Test Your Strategy?
# 
# When you feel like you have an optimization strategy, complete the `promotion_strategy` function to pass to the `test_results` function.  
# From past data, we know there are four possible outomes:
# 
# Table of actual promotion vs. predicted promotion customers:  
# 
# <table>
# <tr><th></th><th colspan = '2'>Actual</th></tr>
# <tr><th>Predicted</th><th>Yes</th><th>No</th></tr>
# <tr><th>Yes</th><td>I</td><td>II</td></tr>
# <tr><th>No</th><td>III</td><td>IV</td></tr>
# </table>
# 
# The metrics are only being compared for the individuals we predict should obtain the promotion – that is, quadrants I and II.  Since the first set of individuals that receive the promotion (in the training set) receive it randomly, we can expect that quadrants I and II will have approximately equivalent participants.  
# 
# Comparing quadrant I to II then gives an idea of how well your promotion strategy will work in the future. 
# 
# Get started by reading in the data below.  See how each variable or combination of variables along with a promotion influences the chance of purchasing.  When you feel like you have a strategy for who should receive a promotion, test your strategy against the test dataset used in the final `test_results` function.

# In[1]:


get_ipython().system('pip3 install --upgrade pip')


# In[2]:


get_ipython().system(' pip3 install wheel')


# In[3]:


get_ipython().system(' pip install xgboost')


# In[4]:


get_ipython().system('pip install --upgrade scikit-learn')


# In[6]:


get_ipython().system(' pip install imblearn')


# In[7]:


# load in packages
#from itertools import combinations

from test_results import test_results, score
import numpy as np
import pandas as pd
import scipy as sp
import xgboost as xgb

#import sklearn as sk

from imblearn.over_sampling import SMOTE
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


import matplotlib.pyplot as plt
import seaborn as sb
get_ipython().run_line_magic('matplotlib', 'inline')

# load in the data
train_data = pd.read_csv('./training.csv')
train_data.head()


# In[8]:


train_data.shape


# In[9]:


train_data.V1.value_counts()


# In[10]:


train_data.V4.value_counts()


# In[11]:


train_data.V5.value_counts()


# In[12]:


train_data.V6.value_counts()


# In[13]:


train_data.V7.value_counts()


# In[14]:


train_data.isnull().sum()


# In[15]:


train_data.duplicated().sum()


# In[16]:


train_data.ID.duplicated().sum()


# In[17]:


train_data.Promotion.value_counts()


# In[18]:


train_data.purchase.value_counts()


# In[19]:


train_data.query(" Promotion == 'Yes' and purchase == 1 ").sort_values(['ID'], ascending = False).reset_index().count()


# In[20]:


train_data.query(" Promotion == 'No' and purchase == 0 ").sort_values(['ID'], ascending = False).reset_index().count()


# In[21]:


train_data.query(" Promotion == 'Yes' and purchase == 0 ").sort_values(['ID'], ascending = False).reset_index().count()


# In[22]:


train_data.query(" Promotion == 'No' and purchase == 1 ").sort_values(['ID'], ascending = False).reset_index().count()


# In[23]:


response = []
for ele, row in train_data.iterrows():
    if row['Promotion'] == 'Yes' and row['purchase'] == 1:
        response.append(1)
    else:
        response.append(0)
        


# In[24]:


train_data['response'] = response


# In[25]:


train_data.head()


# In[26]:


features = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7']
features


# In[27]:


x = train_data[features]
y = train_data['response']


# In[28]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)


# In[29]:


x_train.head()


# In[30]:


x_train.shape, y_train.shape


# In[31]:


from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X, Y = sm.fit_resample(x_train, y_train)


# In[32]:


X.shape, Y.shape


# In[33]:


x_new_train = pd.DataFrame(X, columns = features)
y_new_train = pd.Series(Y)


# In[34]:


x_new_train.head()


# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# x_new_train = sc.fit_transform(x_new_train)
# x_test = sc.transform(x_test)

# ## RandomForestClassifier

# In[35]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 42, n_estimators = 500, max_depth = 5)
rfc.fit(x_new_train, y_new_train)


# In[36]:


def promotion_strategy_2(df):
    '''
    INPUT 
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion - np.array with the values
                   'Yes' or 'No' related to whether or not an 
                   individual should recieve a promotion 
                   should be the length of df.shape[0]
                
    Ex:
    INPUT: df
    
    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2
    
    OUTPUT: promotion
    
    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and 
    the last should not.
    '''
    y_pred = rfc.predict(df)
    
    promotion = []
    for ele in y_pred:
        if ele == 1:
            promotion.append('Yes')
        else:
            promotion.append('No')
            
    
    promotion = np.array(promotion)
    
    return promotion


# In[37]:


promotion_strategy_2(x_test)


# In[38]:


y_pred = rfc.predict(x_test)

print(classification_report(y_pred, y_test))


# In[39]:


confusion_matrix(y_pred, y_test)


# In[40]:


accuracy_score(y_pred, y_test)


# In[41]:


# This will test your results, and provide you back some information 
# on how well your promotion_strategy will work in practice

test_results(promotion_strategy_2)


# ## GradientBoostingClassifier

# In[42]:


from sklearn.ensemble import GradientBoostingClassifier

clf = GradientBoostingClassifier(n_estimators=600, learning_rate=1.0, max_depth=1, random_state=42).fit(x_new_train, y_new_train)


# In[43]:


def promotion_strategy_3(df):
    '''
    INPUT 
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion - np.array with the values
                   'Yes' or 'No' related to whether or not an 
                   individual should recieve a promotion 
                   should be the length of df.shape[0]
                
    Ex:
    INPUT: df
    
    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2
    
    OUTPUT: promotion
    
    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and 
    the last should not.
    '''
    y_pred = clf.predict(df)
    
    promotion = []
    for ele in y_pred:
        if ele == 1:
            promotion.append('Yes')
        else:
            promotion.append('No')
            
    
    promotion = np.array(promotion)
    
    return promotion


# In[44]:


test_results(promotion_strategy_3)


# In[45]:


y_preds = clf.predict(x_test)

print(classification_report(y_preds, y_test))


# In[46]:


confusion_matrix(y_preds, y_test)


# In[47]:


accuracy_score(y_preds, y_test)


# ## xgboostClassifier

# In[48]:


import xgboost as xgb
eval_set = [(x_new_train, y_new_train), (x_test, y_test)]
model = xgb.XGBClassifier(learning_rate = 0.175,
                          max_depth = 7,
                          min_child_weight = 5,
                          objective = 'binary:logistic',
                          seed = 42,
                          gamma = 0.1,
                          silent = True)
model.fit(x_new_train, y_new_train, eval_set=eval_set, eval_metric="auc", verbose=True, early_stopping_rounds=30)


# In[49]:


def promotion_strategy_1(df):
    '''
    INPUT 
    df - a dataframe with *only* the columns V1 - V7 (same as train_data)

    OUTPUT
    promotion - np.array with the values
                   'Yes' or 'No' related to whether or not an 
                   individual should recieve a promotion 
                   should be the length of df.shape[0]
                
    Ex:
    INPUT: df
    
    V1	V2	  V3	V4	V5	V6	V7
    2	30	-1.1	1	1	3	2
    3	32	-0.6	2	3	2	2
    2	30	0.13	1	1	4	2
    
    OUTPUT: promotion
    
    array(['Yes', 'Yes', 'No'])
    indicating the first two users would recieve the promotion and 
    the last should not.
    '''
    y_pred = model.predict(df)
    
    promotion = []
    for ele in y_pred:
        if ele == 1:
            promotion.append('Yes')
        else:
            promotion.append('No')
            
    
    promotion = np.array(promotion)
    
    return promotion


# In[50]:


test_results(promotion_strategy_1)


# In[51]:


yy_pred = model.predict(x_test)

print(classification_report(yy_pred, y_test))


# In[52]:


confusion_matrix(yy_pred, y_test)


# In[53]:


accuracy_score(yy_pred, y_test)


# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# ##Health-insurance risk dataset with 3 numeric columns and 1 categorical, and the target Illness (Yes/No).

# Step 1: Create the dataset (with noise)

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


# Dataset
data = pd.DataFrame({
    "Gender": ["M", "F", "M", "F", "M"],
    "Age": [38, 52, 45, 29, 61],
    "Income": [420000, 360000, 780000, 300000, 500000],
    "Smoking": [0, 5, 0, 12, 8],
    "Illness": ["No", "Yes", "Yes", "Yes", "Yes"]  # Row-3 is noise
})

# Encode target: Yes = +1, No = -1
data["y"] = data["Illness"].map({"Yes": 1, "No": -1})

data


# Step 2: Initialize weights

# In[3]:


n = len(data)
weights = np.ones(n) / n
weights


# ROUND 1 – Smoking Stump

# In[4]:


def h1(smoking):
    return 1 if smoking >= 1 else -1


# In[5]:


pred_h1 = data["Smoking"].apply(h1).values
pred_h1


# Weighted error ε₁

# In[6]:


epsilon1 = np.sum(weights[pred_h1 != data["y"]])
epsilon1


# Alpha α₁

# In[7]:


alpha1 = 0.5 * np.log((1 - epsilon1) / epsilon1)
alpha1


# Update weights

# In[8]:


weights = weights * np.exp(-alpha1 * data["y"] * pred_h1)
weights = weights / np.sum(weights)   # normalize
weights


# ROUND 2 – Age ≥ 45 Stump

# In[9]:


def h2(age):
    return 1 if age >= 45 else -1


# In[10]:


pred_h2 = data["Age"].apply(h2).values
pred_h2


# Weighted error ε₂

# In[11]:


epsilon2 = np.sum(weights[pred_h2 != data["y"]])
epsilon2


# In[12]:


alpha2 = 0.5 * np.log((1 - epsilon2) / epsilon2)
alpha2


# Update weights

# In[13]:


weights = weights * np.exp(-alpha2 * data["y"] * pred_h2)
weights = weights / np.sum(weights)
weights


# ROUND 3 – Smoking Stump Again

# In[14]:


pred_h3 = pred_h1


# Weighted error ε₃

# In[15]:


epsilon3 = np.sum(weights[pred_h3 != data["y"]])
epsilon3


# Alpha α₃

# alpha3 = 0.5 * np.log((1 - epsilon3) / epsilon3)
# alpha3

# In[17]:


alpha3 = 0.5 * np.log((1 - epsilon3) / epsilon3)


# In[18]:


# ROUND 3
pred_h3 = pred_h1  # same smoking stump

epsilon3 = np.sum(weights[pred_h3 != data["y"]])
alpha3 = 0.5 * np.log((1 - epsilon3) / epsilon3)

alpha3


# In[19]:


final_score = (
    (alpha1 + alpha3) * pred_h1 +
    alpha2 * pred_h2
)

final_pred = np.sign(final_score)
final_pred


# ##H.W: Continue with Round-4 and Round-5 using stumps on Income thresholds (e.g., Income ≥ 600000) and show whether the ensemble can eventually flip Row-3 without breaking others.

# In[20]:


import numpy as np
import pandas as pd


data = pd.DataFrame({
    "Age": [38, 52, 45, 29, 61],
    "Income": [420000, 360000, 780000, 300000, 500000],
    "Smoking": [0, 5, 0, 12, 8],
    "y": [-1, 1, 1, 1, 1]  # Row-3 is noisy
})

weights = np.array([0.0714, 0.0714, 0.2857, 0.5, 0.0714])

def h_income(income):
    return 1 if income >= 600000 else -1

pred_h4 = data["Income"].apply(h_income).values

epsilon4 = np.sum(weights[pred_h4 != data["y"]])
alpha4 = 0.5 * np.log((1 - epsilon4) / epsilon4)

weights = weights * np.exp(-alpha4 * data["y"] * pred_h4)
weights = weights / np.sum(weights)

def h_income2(income):
    return 1 if income >= 450000 else -1

pred_h5 = data["Income"].apply(h_income2).values

epsilon5 = np.sum(weights[pred_h5 != data["y"]])
alpha5 = 0.5 * np.log((1 - epsilon5) / epsilon5)

# Predefined stumps from earlier rounds
h_smoke = np.where(data["Smoking"] >= 1, 1, -1)   # h1 & h3
h_age = np.where(data["Age"] >= 45, 1, -1)        # h2

final_score = (
    1.1512 * h_smoke +   # alpha1 + alpha3
    0.9730 * h_age +     # alpha2
    alpha4 * pred_h4 +
    alpha5 * pred_h5
)

final_pred = np.sign(final_score)

results_example1 = data.copy()
results_example1["Final_Prediction"] = final_pred

print("=== Example 1 Final Results ===")
print(results_example1)
print("\nAccuracy:", np.mean(final_pred == data["y"]))


data2 = pd.DataFrame({
    "Gender": ["F", "M", "M", "F", "M"],
    "Age": [33, 57, 41, 49, 36],
    "Income": [480000, 320000, 900000, 540000, 450000],
    "BMI": [22.1, 29.5, 24.0, 31.2, 27.8],
    "Illness": ["No", "Yes", "No", "Yes", "No"]
})

data2["y"] = data2["Illness"].map({"Yes": 1, "No": -1})

n = len(data2)
weights2 = np.ones(n) / n

def h_bmi(bmi):
    return 1 if bmi >= 27 else -1

pred_bmi = data2["BMI"].apply(h_bmi).values

epsilon = np.sum(weights2[pred_bmi != data2["y"]])
alpha = 0.5 * np.log((1 - epsilon) / epsilon)

weights2 = weights2 * np.exp(-alpha * data2["y"] * pred_bmi)
weights2 = weights2 / np.sum(weights2)

results_example2 = data2.copy()
results_example2["BMI_Prediction"] = pred_bmi
results_example2["Updated_Weights"] = weights2

print("\n=== Example 2 Results After 1 Round ===")
print(results_example2)


# In[ ]:





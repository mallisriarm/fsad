# Import Libraries
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()

# Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Models
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier

# Read the data and show first 5 rows
data = pd.read_csv("../input/bs140513_032310.csv")
print(data.head())
print(data.info())

# Create two dataframes with fraud and non-fraud data
df_fraud = data.loc[data.fraud == 1]
df_nonfraud = data.loc[data.fraud == 0]

# Count plot of fraud vs non-fraud
sns.countplot(x="fraud", data=data)
plt.title("Count of Fraudulent Payments")
plt.show()

print("Number of normal examples:", df_nonfraud.fraud.count())
print("Number of fraudulent examples:", df_fraud.fraud.count())

# Mean feature values per category
print("Mean feature values per category:")
print(data.groupby('category')[['amount', 'fraud']].mean())

# Compare Fraud vs Non-Fraud mean amounts per category
comparison = pd.concat(
    [
        df_fraud.groupby('category')['amount'].mean(),
        df_nonfraud.groupby('category')['amount'].mean(),
        data.groupby('category')['fraud'].mean() * 100
    ],
    keys=["Fraudulent", "Non-Fraudulent", "Percent (%)"],
    axis=1,
    sort=False
).sort_values(by=["Non-Fraudulent"])

print(comparison)

# Plot boxplot for amount spent per category
plt.figure(figsize=(30, 10))
sns.boxplot(x=data.category, y=data.amount)
plt.title("Boxplot for the Amount Spent in Category")
plt.ylim(0, 4000)
plt.xticks(rotation=90)
plt.show()

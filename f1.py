# ===========================
# 1. IMPORT LIBRARIES
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# ===========================
# 2. LOAD DATA
# ===========================
df = pd.read_csv("Nutrition__Physical_Activity__and_Obesity.csv", low_memory=False)

# ===========================
# 3. EDA + CLEANING
# ===========================

print("\n===== FIRST 5 ROWS =====")
print(df.head())

print("\n===== SHAPE OF DATA =====")
print(df.shape)

print("\n===== COLUMN NAMES =====")
print(df.columns)

print("\n===== DATA INFO =====")
df.info()

print("\n===== STATISTICAL SUMMARY =====")
print(df.describe())

print("\n===== MISSING VALUES =====")

print(df.isnull().sum())


df = df.dropna(subset=['Data_Value', 'Sample_Size'])

print("\n===== SHAPE AFTER CLEANING =====")

print(df.shape)
# ===========================
# 4. OUTLIER VISUALIZATION
# ===========================

# Histogram
plt.figure(figsize=(8,5))
sns.histplot(df['Data_Value'], kde=True)
plt.title("Histogram of Data Values\nObjective: Understand distribution of Data_Value")
plt.xlabel("Data Value")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

# Boxplot
plt.figure(figsize=(6,8))
sns.boxplot(y=df['Data_Value'])
plt.title("Boxplot of Data Value\nObjective: Detect outliers in Data_Value")
plt.ylabel("Data Value")
plt.grid(alpha=0.3)
plt.show()

# ===========================
# 5. OUTLIER DETECTION (IQR)
# ===========================
Q1 = df['Data_Value'].quantile(0.25)
Q3 = df['Data_Value'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[(df['Data_Value'] >= lower) & (df['Data_Value'] <= upper)]

print("After outlier removal:", df_clean.shape)
# ===========================
# 6. PAIR PLOT
# ===========================
sns.pairplot(df_clean[['Data_Value', 'Sample_Size']])
plt.suptitle("Pair Plot\nObjective: Visualize relationships between variables", y=1.02)
plt.show()

# ===========================
# OBJECTIVE 1 (COUNT PLOT)
# ===========================
# Objective:
# To analyze how the number of records varies across different years.
# This helps in understanding the distribution of data over time.
# It also shows trends or changes in reporting across years.

plt.figure(figsize=(10,5))
sns.countplot(x=df_clean['YearStart'])
plt.title("Count Plot of Records by Year\nObjective: Analyze distribution of records over years")
plt.xlabel("Year")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()

# ===========================
# OBJECTIVE 2 (SCATTER PLOT)
# ===========================
# Objective:
# To examine the relationship between Data_Value and Sample_Size.
# This helps in identifying whether both variables are related.
# It shows how changes in one variable affect the other.

plt.figure(figsize=(8,5))
plt.scatter(df_clean['Data_Value'], df_clean['Sample_Size'], alpha=0.5)
plt.title("Scatter Plot: Data_Value vs Sample_Size\nObjective: Study relationship between variables")
plt.xlabel("Data Value")
plt.ylabel("Sample Size")
plt.grid(alpha=0.3)
plt.show()

# ===========================
# OBJECTIVE 3 (HORIZONTAL BAR CHART)
# ===========================
# Objective:
# To identify the top locations with the highest number of records.
# This helps in finding which locations have more data or activity.
# It highlights areas with higher contribution in the dataset.

top_states = df_clean['LocationDesc'].value_counts().head(10)

plt.figure(figsize=(10,6))
sns.barplot(y=top_states.index, x=top_states.values)

plt.title("Bar Chart: Top 10 Locations\nObjective: Identify locations with highest records")
plt.xlabel("Count")
plt.ylabel("Location")

plt.show()

# ===========================
# OBJECTIVE 4 (PIE CHART)
# ===========================
# Objective:
# To analyze the proportion of top locations in percentage.
# This helps in comparing the relative contribution of each location.
# It gives a quick overview of distribution among top categories.

top5 = df_clean['LocationDesc'].value_counts().head(5)

plt.figure(figsize=(6,6))
plt.pie(top5, labels=top5.index, autopct='%1.1f%%')

plt.title("Pie Chart: Top 5 Locations Distribution\nObjective: Compare proportion of top locations")

plt.show()

# ===========================
# OBJECTIVE 5 (CORRELATION)
# ===========================
# Objective:
# To analyze the relationship between numerical variables.
# This helps in understanding how strongly variables are related.
# It shows positive or negative relationships between features.

num_df = df_clean[['Data_Value', 'Sample_Size']]
corr = num_df.corr()

plt.figure(figsize=(6,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')

plt.title("Heatmap: Correlation Matrix\nObjective: Analyze relationships between numerical variables")

plt.show()

# ===========================
# LINE CHART (TREND ANALYSIS)
# ===========================
# Objective:
# To analyze the trend of Data_Value over different years.
# This helps in identifying patterns such as increase or decrease over time.
# It provides a clear view of changes in data across years.

year_data = df_clean.groupby('YearStart')['Data_Value'].mean()

plt.figure(figsize=(10,5))
plt.plot(year_data.index, year_data.values, marker='o')

plt.title("Line Chart: Data_Value Trend Over Years\nObjective: Analyze trend over time")
plt.xlabel("Year")
plt.ylabel("Average Data Value")

plt.grid(alpha=0.3)
plt.show()

# ===========================
# 7. HYPOTHESIS TEST
# ===========================
# Objective:
# To test whether there is a significant difference between Data_Value and Sample_Size.
# This helps in validating if the observed differences are meaningful.
# It uses statistical testing to support data analysis conclusions.

group1 = df_clean['Data_Value']
group2 = df_clean['Sample_Size']

t_stat, p_value = ttest_ind(group1, group2)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject H0 → Significant Difference")
else:
    print("Fail to Reject H0")

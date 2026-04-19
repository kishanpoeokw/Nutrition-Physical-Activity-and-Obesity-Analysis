import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from scipy.stats import ttest_ind

# loading the dataset
df = pd.read_csv("Nutrition__Physical_Activity__and_Obesity.csv", low_memory=False)

# *** EDA ***

print("\n--- first 5 rows ---")
print(df.head())

print("\n--- last 5 rows ---")
print(df.tail())

print("\n--- shape ---")
print(df.shape)

print("\n--- columns ---")
print(df.columns)

print("\n--- info ---")
df.info()

print("\n--- description ---")
print(df.describe())

print("\n--- missing values ---")
print(df.isnull().sum())

# *** DATA CLEANING ***

# removing extra spaces from column names
df.columns = df.columns.str.strip()

# dropping rows where Data_Value or Sample_Size is missing
df = df.dropna(subset=['Data_Value', 'Sample_Size'])

print("\n--- shape after cleaning ---")
print(df.shape)

# *** OUTLIER VISUALIZATION ***

# histogram to check how Data_Value is distributed
plt.figure(figsize=(8, 5))
sns.histplot(df['Data_Value'], kde=True)
plt.title("Histogram of Data_Value\nObjective: Understand distribution of health indicator values")
plt.xlabel("Data Value")
plt.ylabel("Frequency")
plt.grid(alpha=0.3)
plt.show()

# boxplot to see if there are any extreme outlier values
plt.figure(figsize=(6, 8))
sns.boxplot(y=df['Data_Value'], color='steelblue')
plt.title("Boxplot of Data_Value\nObjective: Detect outliers in health indicator values")
plt.ylabel("Data Value")
plt.grid(alpha=0.3)
plt.show()

# *** OUTLIER REMOVAL USING IQR METHOD ***

Q1 = df['Data_Value'].quantile(0.25)
Q3 = df['Data_Value'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

print("\nLower Bound:", lower)
print("Upper Bound:", upper)

df_clean = df[(df['Data_Value'] >= lower) & (df['Data_Value'] <= upper)]

print("\n--- shape after outlier removal ---")
print(df_clean.shape)

# *** PAIR PLOT ***

sns.pairplot(df_clean[['Data_Value', 'Sample_Size', 'Low_Confidence_Limit', 'High_Confidence_Limit']].sample(500, random_state=42))
plt.suptitle("Pair Plot: Numerical Variables\nObjective: Visualize relationships between numerical variables", y=1.02)
plt.show()

# -------------------------------------------------------
# OBJECTIVE 1 : COUNT PLOT
# I wanted to see which health topic class has the most records in the dataset.
# This gives an idea of which topics were surveyed more across the states.
# -------------------------------------------------------

plt.figure(figsize=(10, 5))
sns.countplot(x='Class', data=df_clean, order=df_clean['Class'].value_counts().index)
plt.title("Count Plot: Records by Health Class\nObjective 1: Analyze frequency of each health topic class")
plt.xlabel("Health Class")
plt.ylabel("Count")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# OBJECTIVE 2 : LINEAR REGRESSION
# I used linear regression to check if Sample_Size can predict Data_Value.
# The regression line shows the trend and how well both variables are related.
# -------------------------------------------------------

X = df_clean[['Sample_Size']]
y = df_clean['Data_Value']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

plt.figure(figsize=(8, 5))
plt.scatter(df_clean['Sample_Size'], df_clean['Data_Value'], alpha=0.4, label="Actual Data")
plt.plot(df_clean['Sample_Size'], y_pred, color='red', label="Regression Line")
plt.title("Linear Regression: Sample_Size vs Data_Value\nObjective 2: Predict Data_Value using Sample_Size")
plt.xlabel("Sample Size")
plt.ylabel("Data Value (%)")
plt.legend()
plt.grid(alpha=0.3)
plt.show()

print("\n--- regression result ---")
print(f"Slope: {model.coef_[0]:.4f}")
print(f"Intercept: {model.intercept_:.4f}")

# -------------------------------------------------------
# OBJECTIVE 3 : LINE GRAPH
# I grouped the data by year and calculated the average Data_Value each year.
# This helps to see if health indicator values like obesity went up or down over time.
# -------------------------------------------------------

year_trend = df_clean.groupby('YearStart')['Data_Value'].mean()

plt.figure(figsize=(10, 5))
plt.plot(year_trend.index, year_trend.values, marker='o', color='tomato')
plt.title("Line Graph: Average Data_Value Over the Years\nObjective 3: Identify yearly trends in health indicator values")
plt.xlabel("Year")
plt.ylabel("Average Data Value (%)")
plt.grid(alpha=0.3)
plt.show()

# -------------------------------------------------------
# OBJECTIVE 4 : CORRELATION HEATMAP
# I calculated correlation between all numerical columns to find relationships.
# High correlation means two variables move together, low means they are not related.
# -------------------------------------------------------

num_df = df_clean[['Data_Value', 'Sample_Size', 'Low_Confidence_Limit', 'High_Confidence_Limit']]
corr = num_df.corr()

print("\n--- correlation matrix ---")
print(corr)

plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Heatmap: Correlation Matrix\nObjective 4: Analyze relationship between numerical variables")
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# OBJECTIVE 5 : PIE CHART
# I took the top 5 states with the most records and plotted their proportion.
# This shows which states contributed the most data to this survey.
# -------------------------------------------------------

top5 = df_clean['LocationDesc'].value_counts().head(5)

plt.figure(figsize=(7, 7))
plt.pie(top5.values, labels=top5.index, autopct='%1.1f%%', startangle=140)
plt.title("Pie Chart: Top 5 States by Record Count\nObjective 5: Compare proportion of records across top states")
plt.show()

# *** HYPOTHESIS TESTING ***
# I divided data into two groups based on sample size and tested if their
# Data_Value averages are significantly different from each other.
# H0: There is no significant difference in Data_Value between both groups.
# H1: There is a significant difference in Data_Value between both groups.

median_sample = df_clean['Sample_Size'].median()

high_sample_group = df_clean[df_clean['Sample_Size'] >= median_sample]['Data_Value']
low_sample_group = df_clean[df_clean['Sample_Size'] < median_sample]['Data_Value']

t_stat, p_value = ttest_ind(high_sample_group, low_sample_group)

print("\n--- hypothesis testing result ---")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value    : {p_value:.4f}")

if p_value < 0.05:
    print("Conclusion: Reject H0 -> There IS a significant difference in Data_Value between high and low sample size groups.")
else:
    print("Conclusion: Fail to Reject H0 -> There is NO significant difference.")

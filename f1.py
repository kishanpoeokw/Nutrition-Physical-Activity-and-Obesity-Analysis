# ===========================
# 1. IMPORT LIBRARIES
# ===========================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import ttest_ind

# ===========================
# 2. LOAD DATA
# ===========================
df = pd.read_csv(
    "Crimes_-_2001_to_Present.csv",
    on_bad_lines='skip',
    low_memory=False
)

print("First 5 rows:\n", df.head())

# ===========================
# 3. EDA
# ===========================
print("\nShape:", df.shape)
print("\nColumns:", df.columns)

print("\nInfo:")
df.info()

print("\nDescription:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# ===========================
# 4. FEATURE ENGINEERING
# ===========================
df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
df = df.dropna(subset=['Date'])

df['Year'] = df['Date'].dt.year
df['Hour'] = df['Date'].dt.hour

# ===========================
# 5. CONTINUOUS DISTRIBUTION
# ===========================
plt.figure(figsize=(10,5))

sns.histplot(
    df['Hour'],
    bins=24,
    kde=True,
    stat="density"
)

plt.title("Continuous Distribution of Crime Occurrence Over Hours", fontsize=14)
plt.xlabel("Hour of the Day (0–23)")
plt.ylabel("Density")
plt.grid(alpha=0.3)
plt.show()

skewness = df['Hour'].skew()
print("Skewness:", skewness)

if skewness == 0:
    print("Normal Data → Z-score")
else:
    print("Skewed Data → IQR Method")

# ===========================
# 6. OUTLIER DETECTION (IQR)
# ===========================
Q1 = df['Hour'].quantile(0.25)
Q3 = df['Hour'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Hour'] < lower) | (df['Hour'] > upper)]
print("Outliers count:", len(outliers))

df_clean = df.copy()

# ===========================
# 7. PAIR PLOT
# ===========================
sns.pairplot(df_clean[['Year', 'Hour']])
plt.show()

# ===========================
# OBJECTIVE 1
# ===========================
# Analyze crime trend over time
plt.figure(figsize=(12,5))
sns.countplot(x='Year', data=df_clean)
plt.xticks(rotation=45)
plt.title("Year-wise Distribution of Crime Incidents", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Number of Crimes")
plt.show()

# ===========================
# OBJECTIVE 2
# ===========================
# Most frequent crime types
plt.figure(figsize=(12,5))
df_clean['Primary Type'].value_counts().head(10).plot(kind='bar')
plt.title("Top 10 Most Frequent Crime Types", fontsize=14)
plt.xlabel("Crime Type")
plt.ylabel("Frequency")
plt.xticks(rotation=45)
plt.show()

# ===========================
# OBJECTIVE 3
# ===========================
# Crime by location
plt.figure(figsize=(12,6))
sns.countplot(
    y='Location Description',
    data=df_clean,
    order=df_clean['Location Description'].value_counts().iloc[:10].index
)
plt.title("Top 10 Locations with Highest Crime Occurrence", fontsize=14)
plt.xlabel("Number of Crimes")
plt.ylabel("Location")
plt.show()

# ===========================
# OBJECTIVE 4 (UPDATED SCATTER)
# ===========================
# Relationship between time and crime occurrence

plt.figure(figsize=(8,5))
plt.scatter(df_clean['Year'], df_clean['Hour'], alpha=0.3)

plt.title("Scatter Plot of Crime Occurrence (Year vs Hour)", fontsize=14)
plt.xlabel("Year")
plt.ylabel("Hour of Crime")
plt.grid(alpha=0.3)

plt.show()

# ===========================
# OBJECTIVE 5 (PIE CHART)
# ===========================
# Proportion of arrests

arrest_counts = df_clean['Arrest'].value_counts()

plt.figure(figsize=(6,6))
plt.pie(
    arrest_counts,
    labels=arrest_counts.index,
    autopct='%1.1f%%',
    startangle=90
)

plt.title("Proportion of Crimes Resulting in Arrests", fontsize=14)
plt.show()

# ===========================
# CORRELATION (IMPROVED)
# ===========================
num_df = df_clean.select_dtypes(include=['int64','float64'])

corr = num_df.corr()

plt.figure(figsize=(8,6))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)

plt.title("Correlation Matrix of Numerical Features", fontsize=14)
plt.show()

# ===========================
# LINEAR REGRESSION
# ===========================
crime_count = df_clean.groupby(['Year', 'Hour']).size().reset_index(name='Crime_Count')

X = crime_count[['Year', 'Hour']]
y = crime_count['Crime_Count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted Crime Count (Linear Regression)", fontsize=14)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.grid(alpha=0.5)
plt.show()

# ===========================
# HYPOTHESIS TEST
# ===========================
group1 = df_clean[df_clean['Arrest'] == True]['Hour']
group2 = df_clean[df_clean['Arrest'] == False]['Hour']

t_stat, p_value = ttest_ind(group1, group2)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("Reject H0 → Significant Difference")
else:
    print("Fail to Reject H0")

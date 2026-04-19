# Nutrition, Physical Activity and Obesity Analysis

A data analysis project on the **USA Behavioral Risk Factor Surveillance System** dataset.
This project explores health indicators like obesity, physical activity and fruit/vegetable consumption
across different US states from 2011 to 2024.

---

## Dataset

- **Source:** CDC Behavioral Risk Factor Surveillance System
- **File:** `Nutrition__Physical_Activity__and_Obesity.csv`
- **Rows:** 1,10,880
- **Columns:** 33
- **Years:** 2011 - 2024
- **Topics Covered:** Obesity / Weight Status, Physical Activity, Fruits and Vegetables

### Key Columns
| Column | Description |
|--------|-------------|
| `Data_Value` | Percentage of adults with that health condition |
| `Sample_Size` | Number of people surveyed |
| `Low_Confidence_Limit` | Lower bound of confidence interval |
| `High_Confidence_Limit` | Upper bound of confidence interval |
| `Class` | Health topic category |
| `LocationDesc` | US State name |
| `YearStart` | Year of survey |

---

## Objectives

| # | Objective | Graph Used |
|---|-----------|------------|
| 1 | Analyze frequency of each health topic class | Count Plot |
| 2 | Predict Data_Value using Sample_Size | Linear Regression |
| 3 | Identify yearly trends in health indicator values | Line Graph |
| 4 | Analyze relationship between numerical variables | Correlation Heatmap |
| 5 | Compare proportion of records across top states | Pie Chart |

---

## Steps Followed

1. **Load Data** - Read CSV using pandas
2. **EDA** - head, tail, shape, info, describe, null values
3. **Data Cleaning** - Dropped rows with missing Data_Value and Sample_Size
4. **Outlier Detection** - Histogram and Boxplot
5. **Outlier Removal** - IQR Method (Lower: 6.15, Upper: 56.15)
6. **Pair Plot** - Visualized relationships between numerical variables
7. **5 Objectives** - Count plot, Linear Regression, Line Graph, Heatmap, Pie Chart
8. **Hypothesis Testing** - T-test between high and low sample size groups

---

## Hypothesis Testing

- **H0:** There is no significant difference in Data_Value between high and low sample size groups
- **H1:** There is a significant difference in Data_Value between high and low sample size groups
- **Result:** T-statistic = -7.2204, P-value = 0.0000
- **Conclusion:** Reject H0 → There IS a significant difference

---

## Libraries Used

```python
pandas
numpy
matplotlib
seaborn
sklearn
scipy
```

---

## How to Run

1. Clone this repository
```
git clone https://github.com/yourusername/your-repo-name.git
```

2. Install required libraries
```
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

3. Place the dataset CSV file in the same folder as the script

4. Run the script
```
python nutrition_final.py
```

---

## Results

- **Physical Activity** has the most records (~49,000) in the dataset
- Average health indicator values **increased from 30.6% in 2011 to 34.6% in 2023**
- **Data_Value and Low_Confidence_Limit** are strongly correlated (0.95)
- **Sample_Size has almost no effect** on Data_Value (correlation = -0.01)
- **National, Maryland, Kansas, Texas and Washington** are the top 5 states by record count

---

## Project Structure

```
📁 project/
├── nutrition_final.py       # main analysis script
├── project.csv              # dataset
└── README.md                # project description
```

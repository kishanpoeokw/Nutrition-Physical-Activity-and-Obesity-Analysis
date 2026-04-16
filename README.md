# 🚔 Crime Data Analysis & Prediction

## 📊 Project Overview

This project performs **exploratory data analysis (EDA), visualization, statistical testing, and machine learning** on a real-world crime dataset.

The goal is to uncover patterns in crime occurrences and build a model to predict crime trends based on time features.

---

## ⚙️ Technologies Used

* Python 🐍
* Pandas & NumPy
* Matplotlib & Seaborn
* Scikit-learn
* SciPy

---

## 📂 Dataset

* Dataset used: *Crimes (2001 to Present)*
* Contains crime records with details like:

  * Date & Time
  * Crime Type
  * Location
  * Arrest status

---

## 🔍 Key Features of Project

### 1. Data Preprocessing

* Handled missing values
* Converted date column to datetime
* Extracted **Year** and **Hour** features

---

### 2. Exploratory Data Analysis (EDA)

* Dataset structure & summary
* Missing value analysis
* Distribution of crime occurrence

---

### 3. Outlier Detection

* Used **IQR (Interquartile Range)** method
* Identified extreme values in crime timing

---

### 4. Data Visualization

* 📈 Year-wise crime trend
* 📊 Top crime types (bar chart)
* 📍 Crime by location
* 🔵 Scatter plot (Year vs Hour)
* 🥧 Arrest proportion (pie chart)
* 🔥 Correlation heatmap

---

### 5. Machine Learning Model

* Model: **Linear Regression**
* Features: Year, Hour
* Target: Crime Count

#### Evaluation Metrics:

* Mean Squared Error (MSE)
* R² Score

---

### 6. Statistical Analysis

* Performed **Hypothesis Testing (t-test)**
* Compared:

  * Crimes with arrest
  * Crimes without arrest

---

## 📊 Results

* Identified trends in crime over years
* Found peak crime hours
* Observed relationships between variables
* Built a predictive model for crime count

---

## 🚀 How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scipy
```

```bash
python f1.py
```

---

## 📌 Future Improvements

* Use advanced models (Random Forest, XGBoost)
* Add real-time data analysis
* Build interactive dashboard (Power BI / Streamlit)

---

## 👨‍💻 Author

**Kishan**
CSE Student | Data Science Enthusiast | Developer

---

## ⭐ If you like this project

Give it a star on GitHub ⭐

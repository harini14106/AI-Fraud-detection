# Credit Card Fraud Detection Project

This project uses Machine Learning models to detect fraudulent credit card transactions. The aim is to accurately classify transactions as fraudulent or legitimate using supervised learning techniques.

## Problem Statement
Credit card fraud is a significant issue in the financial industry. Detecting fraud early helps protect customers and financial institutions, ensuring the safety and trust of digital transactions.

## Dataset
- Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Records: 284,807
- Features: 31 anonymized features (V1-V28) + 'Time', 'Amount', and 'Class'
- Target Variable: `Class` (0 = legitimate, 1 = fraud)

## Project Workflow
1. Data Collection
2. Data Preprocessing (Scaling, SMOTE)
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Model Building (Logistic Regression, Random Forest)
6. Model Evaluation
7. Visualization of Results

## Tools and Technologies
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- imbalanced-learn

## Installation
Clone the repository and install the required libraries:

```bash
pip install -r requirements.txt
```

## How to Run
1. Download the dataset from Kaggle and place it in your working directory.
2. Run the Python source code file.

## Results
- **Random Forest Classifier** performed best with an AUC score of 0.97.
- Top important features: V14, V17, V12.
- SMOTE helped balance the dataset and improved recall.

## Author
- Harini

---

Feel free to fork this project, suggest improvements, or contribute!

---

*Disclaimer: This project is for educational purposes only.*

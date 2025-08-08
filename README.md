# Credit Risk Prediction / Credit Risk Analysis for Extending Bank Loans

## Project Overview
This project focuses on predicting credit risk for bank loan applications. The goal is to build and evaluate several machine learning models to determine whether a customer is likely to default on a loan based on their demographic and financial information. By comparing the performance of different classification algorithms—**Random Forest**, **Support Vector Machine (SVC)**, and **Logistic Regression**—we can identify the most effective model for this prediction task.

---

## Dataset 

[Data Set](/bankloans.csv)

To build the credit risk prediction model, we have used a **bank loan dataset**.
The file `bankloan.csv` contains the information used to create the model.

-   **Rows:** 1150 (700 after cleaning)
-   **Columns:** 9
-   **Target Variable:** `default` (1 = defaulted, 0 = never defaulted)

### Variable Descriptions

| Variable | Description |
| :--- | :--- |
| `age` | Age of the customer in years. |
| `ed` | Educational level of the customer. |
| `employ` | Years with current employer. |
| `address` | Years at current address. |
| `income` | Customer's income in thousands. |
| `debtinc` | Debt to income ratio (%). |
| `creddebt` | Credit to debt ratio. |
| `othdebt` | Other debts. |
| `default` | Customer defaulted in the past (1 = yes, 0 = no). |

---

## Workflow
The project follows these key steps:
**Dataset → Data Cleaning → Exploratory Data Analysis → Preprocessing → Model Training → Evaluation & Comparison**

### 1. Data Cleaning
The initial dataset contained 1150 rows. A check for missing values revealed 450 null entries in the `default` column. These rows were removed, resulting in a clean dataset of 700 instances.

### 2. Exploratory Data Analysis (EDA)
Before modeling, an analysis was performed to understand the data distribution and relationships between variables.

-   **Class Distribution:** The target variable `default` is imbalanced.
    -   **Non-Defaulters (0):** 517 instances
    -   **Defaulters (1):** 183 instances

-   **Debt-to-Income Ratio vs. Age:** A line plot was generated to visualize the relationship between the customer's age and their debt-to-income ratio.

    

### 3. Preprocessing and Modeling
The data was prepared for modeling by splitting it into features (X) and a target (y), followed by scaling and training.

1.  **Train-Test Split:** The dataset was split into training (80%) and testing (20%) sets.
2.  **Feature Scaling:** The features were standardized using `StandardScaler` to ensure all variables contribute equally to model performance.
3.  **Model Training & Evaluation:** Three different classification models were trained and evaluated.

---

## Model Performance

The following models were trained, and their accuracy on the test set was recorded. For the Support Vector Classifier, `GridSearchCV` was used to find the optimal hyperparameters.

| Model | Details | Test Accuracy |
| :--- | :--- | :--- |
| **Random Forest** | `n_estimators=200` | 80.0% |
| **Support Vector Classifier (SVC)** | Default parameters | 79.3% |
| **SVC (Tuned)** | `C=0.1`, `gamma=0.1`, `kernel='linear'` | 82.1% |
| **Logistic Regression** | Default parameters | **83.6%** |

### Results
The **Logistic Regression** model achieved the highest accuracy of **83.6%** on the test set, making it the best-performing model for this task.

The confusion matrix for the Logistic Regression model visualizes its performance in distinguishing between defaulters and non-defaulters.



## How to Run
1.  Clone the repository.
2.  Ensure you have the required libraries installed (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`).
3.  Place the `bankloans.csv` file in the same directory or update the file path in the notebook.
4.  Open and run the `credit_risk.ipynb` notebook in a Jupyter environment.

## Libraries Used
-   `pandas`
-   `numpy`
-   `scikit-learn`
-   `matplotlib`
-   `seaborn`

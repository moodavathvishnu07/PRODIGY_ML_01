# Prodigy InfoTech - Machine Learning Internship

## 🏠 Task-01: House Price Prediction using Linear Regression

![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

### 📌 Problem Statement
> **Task-01:** Implement a linear regression model to predict the prices of houses based on their **square footage** and the number of **bedrooms** and **bathrooms**.
> 
> **Dataset Reference:** [Kaggle: House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

---

### 📁 Project Structure

```
PRODIGY_ML_01/
├── data/
│   └── train.csv                      # Kaggle Ames Housing Dataset
├── models/
│   └── linear_regression_model.joblib # Serialized trained Linear Regression model
├── plots/                             # Publication-quality evaluation plots
│   ├── 1_actual_vs_predicted.png      # Actual vs. Predicted scatter plot with identity line
│   ├── 2_residuals_distribution.png   # Residual error histogram with normal curve
│   ├── 3_correlation_heatmap.png      # Correlation matrix of features with SalePrice
│   ├── 4_feature_coefficients.png     # Linear regression coefficient bar chart
│   └── 5_price_vs_sqft_scatter.png    # Living area vs Price scatter & trendline
├── dataset.py                         # Dataset generator and loader module
├── train_model.py                     # Complete training, evaluation & plotting pipeline
├── predict.py                         # Interactive CLI tool for predicting custom house prices
├── task_01_house_price.ipynb          # Clean, annotated Jupyter Notebook
├── requirements.txt                   # Python environment dependencies
└── README.md                          # Internship documentation and report
```

---

### 📐 Mathematical Formulation

Multiple Linear Regression models the linear relationship between the dependent target variable ($Y$) and multiple independent explanatory variables ($X_1, X_2, \dots, X_n$):

$$\hat{Y} = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \beta_3 X_3 + \beta_4 X_4 + \epsilon$$

Where:
- $\hat{Y}$ = Predicted **SalePrice** (\$)
- $\beta_0$ = Intercept (\$46,166.71)
- $X_1$ = **GrLivArea** (Above grade ground living area in sq ft)
- $X_2$ = **BedroomAbvGr** (Bedrooms above grade)
- $X_3$ = **FullBath** (Full bathrooms above grade)
- $X_4$ = **HalfBath** (Half bathrooms above grade)

#### Learned Regression Equation:
$$\text{SalePrice} = \$46,166.71 + (114.81 \times \text{GrLivArea}) + (9,896.13 \times \text{BedroomAbvGr}) + (11,543.41 \times \text{FullBath}) + (5,950.48 \times \text{HalfBath})$$

---

### 📊 Model Performance & Metrics

The model was trained on **80% of the dataset (1,168 samples)** and evaluated on **20% held-out test data (292 samples)**:

| Metric | Value | Description |
| :--- | :--- | :--- |
| **Training $R^2$ Score** | **0.8313 (83.13%)** | Proportion of variance explained on training set |
| **Testing $R^2$ Score** | **0.7880 (78.80%)** | Proportion of variance explained on unseen test set |
| **Mean Absolute Error (MAE)** | **\$27,366.33** | Average magnitude of absolute prediction error |
| **Root Mean Squared Error (RMSE)** | **\$34,558.95** | Penalized measure of large prediction errors |
| **Mean Absolute % Error (MAPE)** | **10.83%** | Average percentage error across test homes |

---

### 📈 Generated Visualizations

| Visualization | Description |
| :--- | :--- |
| **`plots/1_actual_vs_predicted.png`** | Demonstrates strong linear alignment between ground truth and predicted house prices along the $y = x$ reference line. |
| **`plots/2_residuals_distribution.png`** | Residual errors are centered symmetrically around 0, confirming the normality assumption of Ordinary Least Squares regression. |
| **`plots/3_correlation_heatmap.png`** | Heatmap highlighting that square footage (`GrLivArea`) has the highest single correlation with sale price. |
| **`plots/4_feature_coefficients.png`** | Quantifies the dollar value added per additional square foot (\$114.81/sqft), full bath (+\$11.5k), and bedroom (+\$9.8k). |
| **`plots/5_price_vs_sqft_scatter.png`** | Scatter plot of living area vs sale price with the fitted linear regression line. |

---

### 🚀 How to Run the Project

#### 1. Setup Environment
Ensure Python 3.8+ is installed, then install requirements:
```bash
pip install -r requirements.txt
```

#### 2. Train and Evaluate the Model
Run the end-to-end training pipeline to reproduce metrics and regenerate all plots:
```bash
python train_model.py
```

#### 3. Predict Prices Interactively
Launch the interactive house price prediction CLI tool:
```bash
python predict.py
```
**Example CLI Interaction:**
```
============================================================
      🏠 PRODIGY INFOTECH - HOUSE PRICE ESTIMATOR 🏠
============================================================
Enter Living Area Square Footage (e.g. 1850): 2200
Enter Number of Bedrooms (e.g. 3): 3
Enter Number of Full Bathrooms (e.g. 2): 2
Enter Number of Half Bathrooms [Default: 0]: 1

------------------------------------------------------------
  📊 PROPERTY SPECIFICATIONS:
     - Living Area : 2,200 sq ft
     - Bedrooms    : 3
     - Full Baths  : 2
     - Half Baths  : 1
------------------------------------------------------------
  💰 ESTIMATED HOUSE SALE PRICE: $357,511.45
============================================================
```

#### 4. Run the Jupyter Notebook
Explore the interactive step-by-step notebook:
```bash
jupyter notebook task_01_house_price.ipynb
```

---

### 💡 Key Takeaways
1. **Living Area (Square Footage)** is the dominant predictive driver of residential home prices, contributing approximately **\$114.81 per additional square foot**.
2. **Bathrooms and Bedrooms** provide additive value, with full bathrooms contributing **~\$11,543** per additional bathroom.
3. The Ordinary Least Squares (OLS) Linear Regression model achieved a robust **$R^2$ of ~79%** on unseen test data using solely these core physical dimensions.

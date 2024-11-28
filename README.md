INSURANCE FRAUD DETECTION

This project implements a comprehensive machine learning pipeline to detect insurance fraud. Fraud detection in the insurance industry is critical to minimize financial losses and maintain credibility. The dataset includes policyholder information, claim details, and indicators of fraudulent activity.

Table of Contents
1. [Overview](#overview)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Data Preprocessing](#data-preprocessing)
5. [Modeling and Evaluation](#modeling-and-evaluation)
6. [Results](#results)
7. [Conclusion](#conclusion)

---

 Overview

This project utilizes various machine learning models to detect potential fraud in insurance claims. The models are compared based on their performance metrics, and the best model is highlighted for practical use.

---

Features
- Data Preprocessing:
  - Missing value handling.
  - Feature encoding.
  - Outlier detection and treatment using StandardScaler.
  - Feature correlation analysis.
  
- Machine Learning Models:
  - Support Vector Classifier (SVC)
  - Decision Tree Classifier
  - Random Forest Classifier
  - AdaBoost Classifier
  - XGBoost Classifier
  
- Evaluation Metrics:
  - Accuracy
  - Confusion Matrix
  - Precision, Recall, F1-Score

---

Dependencies

Install the required Python libraries using:
```bash
pip install -r requirements.txt
```

Main Dependencies:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `xgboost`

---

Data Preprocessing

1. Handling Missing Values:
   - Missing values in `collision_type`, `property_damage`, and `police_report_available` were filled with their respective mode.
   - Missing columns were identified and visualized using bar plots.

2. Feature Engineering:
   - Categorical columns were one-hot encoded.
   - Numerical features were standardized using `StandardScaler`.

3. Feature Selection:
   - High correlation was detected among some columns. Columns like `age` and `total_claim_amount` were dropped to improve the model's performance.

4. Outlier Detection:
   - Box plots were generated for all numerical features to identify outliers.

---

Modeling and Evaluation

Models Implemented:
- Support Vector Classifier (SVC):
  - Training Accuracy: 82.8%
  - Testing Accuracy: 77.6%

- Decision Tree Classifier:
  - Training Accuracy: 80.6%
  - Testing Accuracy: 84.4% (after hyperparameter tuning)

- Random Forest Classifier:
  - Training Accuracy: 100%
  - Testing Accuracy: 82.4%

- AdaBoost Classifier:
  - Training Accuracy: 80.6%
  - Testing Accuracy: 84.4%

- XGBoost Classifier:
  - Training Accuracy: 80.6%
  - Testing Accuracy: 84.4%

- Extra Trees Classifier:
  - Training Accuracy: 100%
  - Testing Accuracy: 83.2%

Visualizations:
- Correlation Heatmap: Analyzed relationships between features.
- Model Comparison Bar Plot: Compared the performance of all models based on accuracy.


Results

The Decision Tree Classifier achieved the highest testing accuracy of 84.4%, making it the best model for this project. The detailed comparison of models is shown below:

| Model                     | Accuracy |
|---------------------------|----------|
| Decision Tree             | 84.4%    |
| AdaBoost                  | 84.4%    |
| XGBoost                   | 84.4%    |
| Extra Trees               | 83.2%    |
| Random Forest             | 82.4%    |

Conclusion

The Insurance Fraud Detection project demonstrates a comprehensive approach to identifying fraudulent insurance claims using various machine learning algorithms. By implementing a robust data preprocessing pipeline and evaluating multiple models, the project successfully highlights the importance of feature engineering, correlation analysis, and model tuning in achieving high accuracy.

The  Decision Tree Classifier emerged as the best-performing model with a testing accuracy of **84.4%**, demonstrating the strength of ensemble learning in combining the predictions of multiple models for improved performance. Other models, such as AdaBoost, and XGBoost, also performed well, further showcasing the reliability of tree-based algorithms for classification tasks in this domain.

This project serves as a solid foundation for fraud detection systems, providing insights into best practices for preprocessing, feature selection, and model evaluation. Future enhancements could include:
- Incorporating additional data sources.
- Exploring advanced deep learning techniques.
- Conducting hyperparameter tuning for further model optimization.

Overall, this project underscores the value of machine learning in automating and improving fraud detection processes, helping to reduce financial losses and maintain the integrity of insurance systems.




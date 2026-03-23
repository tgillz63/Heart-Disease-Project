## Heart Disease Prediction Machine Learning Project 

# Overview
Heart disease is the leading cause of death globally, accounting for almost 32% of all deaths. This project uses the 2022 BRFSS (Behavioral Risk Factor Surveillance System) dataset to build a predictive model that predicts heart disease. The project also use SHAP feature value importance to attempt to identify significant lifestyle and medical predictors of heart disease. 

#Methods 
1. **Preprocessing** -Encoded categorical features, scaled numeric features in logistic regression to normalize ranges of these features, converted age category from a range to midpoint of the range to make the feature easier to digest for models.
2. **Handling Class Imbalance** - Given the 5.46% minority class, adjustments had to be made to account for the imbalance. For the random forest I used class_weight='balanced' which automatically calculates class weights inversely proportional to class frequency. For the XGBoost I did essentially the same thing but calculated the scale_pos_weight manually and then multiplied it by 0.5 to dampen what was orginally too agressive of a correction. I also evaluated models across multiple probability thresholds (0.10 to 0.50) to find the optimal balance between Precision and Recall. Due to the class imbalance simple accuracy doesn't cut it for a performance metric, so others that prioritize identifying the minority class were used (recall, percision, F-1 score). 
```
## base random forrest
rf = RandomForestClassifier(class_weight='balanced',random_state=45, verbose=2 )
```
```
## damped scaled weight
scaled_weight=(y_train_f==0).sum()/(y_train_f==1).sum()
scaled_weight=scaled_weight*0.5
```
3. **Logistic Regression** - Helpful for a baseline but not very effective. Recall- 24%
4. **Random Forest** - Started with a base random forest, then used RandomizedSearchCV to tune for the following hyperparameters:'n_estimators', 'max_depth', 'min_samples_split','min_samples_leaf','max_features','max_samples', 'bootstrap'
- Recall 60%, ROC-AUC 88.8%
5. **XGBoost** - Began with base XGBoost, tuned the following hyperparameters with GridsearchCV: 'n_estimators', 'eta'
- Tuned these hyperparameters with RadomizedSearchCV:  "gamma", "max_depth", "min_child_weight",  "subsample", "colsample_bytree"
- Recall 66.7%, ROC-AUC 88.7%

# Insights and Recommendations
- Using SHAP values the features most correlated with heart disease are Angina(chest pain or discomfort occurring when heart muscle receives insufficient oxygenated blood) history, old age, patients who had difficulty walking, current/former smokers, as well as stroke or diabetes history.

<img width="490" height="413" alt="Screenshot 2026-03-23 at 7 43 32 PM" src="https://github.com/user-attachments/assets/85d79db8-73d9-4c8d-9105-eae2713753de" />

-Aggressive management of cholesterol and blood pressure to avoid Angina.
-Increased screening frequency for diabetic patients and those over age 65.
-Lifestyle interventions focusing on smoking cessation, as former smokers still carry significant residual risk.

# Neccesary Installs and Usage
-Installs: pandas, numpy, matplotlib, seaborn, plotnine, scikit-learn, xgboost, shap, statsmodels, imbalanced-learn
- Run python script to see in depth results (NOTE runtimes may be longer than expected due to hyperparameter tuning)
  






  

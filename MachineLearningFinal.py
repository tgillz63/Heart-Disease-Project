
from plotnine import *
import pandas as pd  
import numpy as np   
import xgboost as xgb 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit,RandomizedSearchCV, GridSearchCV , train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.model_selection import ParameterGrid ,StratifiedKFold
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.model_selection import cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import average_precision_score


heart_data=pd.read_csv("~/Downloads/archive (18)/2022/heart_2022_no_nans.csv") 


##organize columns for encoding 
cat_cols = [
   'Sex','GeneralHealth', 'LastCheckupTime',  'RemovedTeeth', 'SmokerStatus', 'ECigaretteUsage'
,'RaceEthnicityCategory', 'TetanusLast10Tdap', 'HadDiabetes', 'CovidPos', "State"
]

numeric_cols=['HeightInMeters','WeightInKilograms','BMI','SleepHours','MentalHealthDays','PhysicalHealthDays']
bin_cols = [
    'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
    'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis',
     'DeafOrHardOfHearing', 'BlindOrVisionDifficulty',  'HadAngina',
    'PhysicalActivities', 'ChestScan', 'AlcoholDrinkers',
    'HIVTesting', 'FluVaxLast12', 'PneumoVaxEver',
    'HighRiskLastYear', 'DifficultyConcentrating',
    'DifficultyWalking', 'DifficultyDressingBathing',
    'DifficultyErrands', 'HadHeartAttack'
]
heart_data['HadAngina'].value_counts()

##heart_data=heart_data.drop(columns=['State'],axis=1)
heart_data_fb=heart_data.copy()
heart_data_fb[bin_cols]=heart_data_fb[bin_cols].replace({'Yes': 1, 'No': 0})

dummies=pd.get_dummies(heart_data_fb[cat_cols],drop_first=False,dtype=int)
heart_data_fb=pd.concat([heart_data_fb.drop(columns=cat_cols),dummies],axis=1)
heart_data_fb['ChestScan']
##Set age as midpoint of range for random forrest then logistic regression
heart_data_fb['AgeCategory']=heart_data_fb['AgeCategory'].str.split().str[1].astype(float) 
heart_data_fb['AgeCategory']=heart_data_fb['AgeCategory']+2
heart_data_fb['AgeCategory']
 


## create dummies and merge with rest of features for logistic regression
heart_data_log=heart_data.copy()
heart_data_log[bin_cols]=heart_data_log[bin_cols].replace({'Yes': 1, 'No': 0})
x_dum=pd.get_dummies(heart_data_log[cat_cols],drop_first=True,dtype=int)
x=pd.concat([heart_data_log.drop(columns=cat_cols),x_dum],axis=1)
x['AgeCategory']=x['AgeCategory'].str.split().str[1].astype(float) 
x['AgeCategory']=x['AgeCategory']+2
x=x.drop('HadHeartAttack', axis=1)

#lgistic regression

x_log=sm.add_constant(x)
y=heart_data_log['HadHeartAttack']

x_train_l, x_test_l, y_train_l, y_test_l=train_test_split(x_log,y,test_size=0.2, stratify=y, random_state=45)


scaler = StandardScaler()


numeric_cols = ['HeightInMeters', 'WeightInKilograms', 'BMI', 'SleepHours', 'MentalHealthDays', 'PhysicalHealthDays', 'AgeCategory']


x_train_l[numeric_cols] = scaler.fit_transform(x_train_l[numeric_cols])
x_test_l[numeric_cols] = scaler.transform(x_test_l[numeric_cols])

log_reg = sm.Logit(y_train_l, x_train_l).fit()
print(log_reg.summary2())
prob1= log_reg.predict(x_test_l)

y_predicted = (prob1 >= 0.5).astype(int) 
print("\nConfusion matrix:")
cm = (confusion_matrix(y_test_l, y_predicted))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)# Set class labels
disp.plot(cmap="Blues") # Set color map
plt.title("Confusion Matrix — XGBoost") # Set title
plt.show() # Display plot
print("\nAccuracy):")



## select x and y columns for random forrest
x_for=heart_data_fb.drop(columns=['HadHeartAttack'], axis=1)
y_for=heart_data_fb['HadHeartAttack']
x_train_f, x_test_f, y_train_f, y_test_f=train_test_split(x_for,y_for,test_size=0.2, stratify=y_for, random_state=45)


## base random forrest
rf = RandomForestClassifier(class_weight='balanced',random_state=45, verbose=2 )

rf.fit(x_train_f,y_train_f)
y_prob= rf.predict_proba(x_test_f,)[:,1]


y_predicted = (y_prob >= 0.2).astype(int)
print(classification_report(y_test_f, y_predicted))

for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, .35, 0.40, 0.45, 0.50]:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    print(f"\n--- Threshold: {threshold} ---")
    print(classification_report(y_test_f, y_pred_threshold))

print(f"ROC-AUC: {roc_auc_score(y_test_f, y_prob):.3f}")
print(f"PR-AUC: {average_precision_score(y_test_f, y_prob):.3f}")

print("\nConfusion matrix:")
cm = (confusion_matrix(y_test_f, y_predicted))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)# Set class labels
disp.plot(cmap="Blues") # Set color map
plt.title("Confusion Matrix — XGBoost") # Set title
plt.show() # Display plot
print("\nAccuracy):")

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix 


##random forrest tuned

cv_strat = StratifiedKFold(n_splits=3, shuffle=True, random_state=45)
params = {
    'n_estimators': [300, 500, 700],           
    'max_depth': [10, 15, 20, 25],             #
    'min_samples_split': [10, 20, 30, 50],     
    'min_samples_leaf': [5, 10, 15, 20],       
    'max_features': ['sqrt', 'log2'],          #
    'max_samples': [0.7, 0.8, 0.9],            
    'bootstrap': [True],
}

tuned_rf=RandomForestClassifier(class_weight='balanced',bootstrap=True, random_state=45, n_jobs=-1,verbose=2)
search= RandomizedSearchCV(estimator=tuned_rf,param_distributions=params, scoring='f1',n_jobs=-1,cv=cv_strat,verbose=3,random_state=45)

search.fit(x_train_f,y_train_f)
print(search.best_params_)


##tuned hyperparameters
best_estimators=500
best_sample_split=10
best_leaf_samples=5
best_max_samples=0.7
best_max_features='sqrt'
best_max_depth=20
bbootstrap=True


best_model = search.best_estimator_
y_prob = best_model.predict_proba(x_test_f)[:, 1]


for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, .35, 0.40, 0.45, 0.50]:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    print(f"\n--- Threshold: {threshold} ---")
    print(classification_report(y_test_f, y_pred_threshold))

y_predicted = (y_prob >= 0.45).astype(int)

print(classification_report(y_test_f, y_predicted))
auc = roc_auc_score(y_test_f, y_prob)
print(f"AUC-ROC: {auc:.3f}")

print("\nConfusion matrix:")
cm = (confusion_matrix(y_test_f, y_predicted))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)# Set class labels
disp.plot(cmap="Blues") # Set color map
plt.title("Confusion Matrix — XGBoost") # Set title
plt.show() # Display plot
print("\nAccuracy):")



## all states equally important
states = x_train_f.columns[-54:] 
States_subset = x_train_f[states]
importances= best_model.feature_importances_
imp_subset = importances[-54]


feature_importance_df = pd.DataFrame({
    'Feature': states,
    'Importance': imp_subset
})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


##XGboost base
## damped scaled weight
scaled_weight=(y_train_f==0).sum()/(y_train_f==1).sum()
scaled_weight=scaled_weight*0.5


dtrain = xgb.DMatrix(data=x_train_f, label=y_train_f)
dtest  = xgb.DMatrix(data=x_test_f,  label=y_test_f)


params = {
        "objective": "binary:logistic", # Set objective
        "eval_metric": ["aucpr", "logloss"],  # Track both AUC and error
        "seed": 42, # set seed
        'scale_pos_weight':scaled_weight,
        'eta': 0.1


    }
num_boost_round = 300# Set number of rounds

watchlist = [(dtrain, "train")] 
base_xgb = xgb.train(params, 
                    dtrain,  
                    num_boost_round=num_boost_round, 
                    evals=watchlist,  
                 
                    verbose_eval=50)




test_pred = base_xgb.predict(dtest) # Create predictions



for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, .35, 0.40, 0.45, 0.50]:
    y_pred_threshold = (test_pred >= threshold).astype(int)
    print(f"\n--- Threshold: {threshold} ---")
    print(classification_report(y_test_f, y_pred_threshold))
#
test_pred_cls = (test_pred >= 0.5).astype(int)

print("\nConfusion matrix:")
cm = (confusion_matrix(y_test_f, test_pred_cls))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)# Set class labels
disp.plot(cmap="Blues") # Set color map
plt.title("Confusion Matrix — XGBoost") # Set title
plt.show() # Display plot
print("\nAccuracy):")
print(classification_report(y_test_f, test_pred_cls))



## XGBoost tuning


treegrid = {
    'n_estimators': [ 300, 500, 700, 1000],
    'eta':  [0.2, 0.1, 0.05, 0.01]
   
}
xgb_model_t = xgb.XGBClassifier( scale_pos_weight=scaled_weight, objective='binary:logistic',
                              eval_metric=['auc', 'error'],  
                              tree_method= "hist",nthread= 1, random_state=45)
xgb_random_t = GridSearchCV(estimator=xgb_model_t, param_grid=treegrid,
                                scoring='f1', n_jobs=-1, cv=cv_strat, verbose=3)
xgb_random_t.fit(x_train_f, y_train_f)
print(xgb_random_t.best_params_)
best_estimators=300
best_eta=0.01
grid1 = {
    "gamma": [0,3,5,9],
     "max_depth": [3,  5, 7],
    "min_child_weight": [ 3, 5, 9, 12]
    }

xgb_model1 = xgb.XGBClassifier(scale_pos_weight=scaled_weight, objective='binary:logistic', 
                               n_estimators=best_estimators, eta=best_eta,
                              eval_metric=['auc', 'error'],  tree_method= "hist",nthread= 1, random_state=45)

xgb_random = RandomizedSearchCV(estimator=xgb_model1, param_distributions=grid1, 
                                scoring='f1', n_jobs=-1, cv=cv_strat, verbose=3)
xgb_random.fit(x_train_f, y_train_f)
print(xgb_random.best_params_)
best_max_depth=3
best_child_weight=3
best_gamma=5

grid2 = {
 "subsample":[0.4, 0.5 ,0.6, 0.7, 0.9,],
"colsample_bytree": [0.4, 0.5, 0.6, 0.7, 0.9,],
    }

xgb_model2 = xgb.XGBClassifier(scale_pos_weight=scaled_weight, n_estimators=best_estimators, 
                               objective='binary:logistic',
                              eval_metric=['auc', 'error'], tree_method= "hist",nthread= 1,
                                random_state=45,max_depth=best_max_depth, min_child_weight=best_child_weight,
                                  gamma=best_gamma)
xgb_random2 = RandomizedSearchCV(estimator=xgb_model2, param_distributions=grid2,
                                scoring='f1', n_jobs=-1, cv=cv_strat, verbose=3)

xgb_random2.fit(x_train_f, y_train_f)
print(xgb_random2.best_params_)
best_subsample=0.9
best_colsample=0.6


tuned_params = { 
    'n_estimators': best_estimators,
    'eta': best_eta,
    'gamma': best_gamma,
    'subsample': best_subsample,
    'colsample_bytree': best_colsample,
    "max_depth": best_max_depth,
    "min_child_weight": best_child_weight,
     'scale_pos_weight': scaled_weight, 
     # Set scale pos weight
       "objective": "binary:logistic", # Set objective
       "eval_metric": ["auc", "error"]
}
xgb_tuned = xgb.train(
    tuned_params,
    dtrain,
    num_boost_round=best_estimators,
             evals=watchlist,
    verbose_eval=50
 )    
test_pred2 = xgb_tuned.predict(dtest)

#
test_pred_cls2 = (test_pred2 >= 0.40).astype(int)
auc = roc_auc_score(y_test_f, test_pred2)
print(f"AUC-ROC: {auc:.3f}")


test_pred2 = xgb_tuned.predict(dtest) 

auc_val = roc_auc_score(y_test_f, test_pred2)
print(f"AUC-ROC: {auc_val:.3f}")


test_pred_cls2 = (test_pred2 >= 0.40).astype(int)
print(classification_report(y_test_f, test_pred_cls2))


print("\nConfusion matrix:")
cm = (confusion_matrix(y_test_f, test_pred_cls2))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap="Blues") 
plt.title("Confusion Matrix — XGBoost") 
plt.show() 
print("\nAccuracy):")
print(classification_report(y_test_f, test_pred_cls2))
for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, .35, 0.40, 0.45, 0.50]:
    y_pred_threshold = (test_pred2 >= threshold).astype(int)
    print(f"\n--- Threshold: {threshold} ---")
    print(classification_report(y_test_f, y_pred_threshold))



y_prob_rf = best_model.predict_proba(x_test_f)[:, 1]
y_prob_xgb = xgb_tuned.predict(dtest)  

##percisoon recall curve
from sklearn.metrics import precision_recall_curve

precision_rf, recall_rf, thresholds = precision_recall_curve(y_test_f, y_prob_rf)
precision_xgb, recall_xgb, thresholds = precision_recall_curve(y_test_f, y_prob_xgb)

avg_precision_xgb = average_precision_score(y_test_f, y_prob_xgb)
avg_precision_rf = average_precision_score(y_test_f, y_prob_rf)


rf_prc = (
    ggplot()
    + geom_line(aes(x=recall_rf, y=precision_rf), color='blue')
    +geom_line(aes(x=recall_xgb, y=precision_xgb), color='red')
    + labs(title=f'Random Forest Precision-Recall Curve (AP = {avg_precision_rf:.3f})', x='Recall', y='Precision')
    + theme_minimal()
)
rf_prc.show()


# Calculate ROC curves
from sklearn import metrics
fpr_rf, tpr_rf, _ = roc_curve(y_test_f, y_prob_rf)
roc_auc_rf = metrics.auc(fpr_rf, tpr_rf)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test_f, y_prob_xgb)
roc_auc_xgb = metrics.auc(fpr_xgb, tpr_xgb)

roc_crv=(
     ggplot()

     +geom_line(aes(x=fpr_rf, y=tpr_rf), color='blue')
     +geom_line(aes(x=fpr_xgb, y=tpr_xgb), color='red')
    +labs(title=f'ROC Curve (AUCs = {roc_auc_rf:.3f}, {roc_auc_xgb:.3f})', x='False Positive Rate', y='True Positive Rate')
     +theme_minimal()    
 )
roc_crv.show()

import shap

shap_test_subset = x_test_f.sample(500, random_state=45) 
explainer = shap.TreeExplainer(xgb_tuned)


dtest_shap = xgb.DMatrix(shap_test_subset)
shap_values = explainer(dtest_shap)


shap_values.feature_names = list(shap_test_subset.columns)
shap_values.data = shap_test_subset.values 

plt.figure(figsize=(10, 8))
shap.plots.bar(shap_values, max_display=25, show=False)
plt.title("Global Feature Importance (XGBoost)")
plt.show()


plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=25, show=False)
plt.show()


plt.figure(figsize=(10, 8))
shap.plots.waterfall(shap_values[0], max_display=25)
plt.show()
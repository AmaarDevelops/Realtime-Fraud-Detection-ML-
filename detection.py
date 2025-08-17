import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score,roc_curve,auc
from imblearn.over_sampling import SMOTE,ADASYN
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('creditcard.csv',encoding='latin1')

missing_values = df.isnull().sum()

x = df.drop(columns=['Class'],axis=1)
y = df['Class']


x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.3,stratify=y)

numerical_features = x_train.columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ('num_scaler',StandardScaler(),numerical_features)
],remainder='passthrough')

cv_splitter = StratifiedKFold(n_splits=3,shuffle=True,random_state=42)


# Model : 1 -- Logistic Regressor :-

pipeline = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',LogisticRegression(random_state=42,solver='liblinear'))
])

param_grid_lr = {
    'classifier__C' : [0.001,0.01,0.1,1,10,100],
    'classifier__class_weight' : [None,'balanced']
}

grid_search_lr = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid_lr,
    cv=cv_splitter,
    scoring='roc_auc',
    n_jobs = -1,
    verbose=2
)

grid_search_lr.fit(x_train,y_train)

print('\n Best params for LR',grid_search_lr.best_params_)
print('Best Cross validated ROC AUC for LR :-', grid_search_lr.best_score_)

#Get the best Model
best_lr_model = grid_search_lr.best_estimator_

y_pred_best_lr = best_lr_model.predict(x_test)




print('\n Model Evaluation LR')
print('Classification Report LR :-', classification_report(y_test,y_pred_best_lr))
print('Confusion matrix LR :-', confusion_matrix(y_test,y_pred_best_lr))
print('\n ROC_AUC_Score LR :-', roc_auc_score(y_test,best_lr_model.predict_proba(x_test)[:,1]))

#Model 2 : Logistic Regression with SMOTE
pipeline_lr_smote = ImbPipeline(steps=[
    ('preprocessor',preprocessor),
    ('smote',SMOTE(random_state=42)),
    ('classifier',LogisticRegression(random_state=42,solver='liblinear'))
])

pipeline_lr_smote.fit(x_train,y_train)

y_pred_lr_smote = pipeline_lr_smote.predict(x_test)

print("\n Model Evaluation LR SMOTE")
print("\n Confusion Matrix LR Smote :-", confusion_matrix(y_test,y_pred_lr_smote))
print('Classification report LR Smote :-', classification_report(y_test,y_pred_lr_smote))
print("ROC_AUC Score LR Smote :-", roc_auc_score(y_test,pipeline_lr_smote.predict_proba(x_test)[:,1]))

#Model 3 : Logistic Regression with ADASYN

pipeline_lr_adasyn = ImbPipeline(steps=[
    ('preprocessor',preprocessor),
    ('adasyn',ADASYN(random_state=42)),
    ('classifier', LogisticRegression(random_state=42,solver='liblinear'))
])

pipeline_lr_adasyn.fit(x_train,y_train)

y_pred_lr_adasyn = pipeline_lr_adasyn.predict(x_test)


print("\n Model Evaluation LR ADASYN")
print("\n Confusion Matrix LR ADASYN :-", confusion_matrix(y_test,y_pred_lr_adasyn))
print('Classification report LR ADASYN :-', classification_report(y_test,y_pred_lr_adasyn))
print("ROC_AUC Score LR ADASYN :-", roc_auc_score(y_test,pipeline_lr_adasyn.predict_proba(x_test)[:,1]))



#Model 4 : XG Boost

pipeline_xg = Pipeline(steps=[
    ('preprocessor',preprocessor),
    ('classifier',XGBClassifier(n_estimators=5))
])

pipeline_xg.fit(x_train,y_train)
y_pred_xg = pipeline_xg.predict(x_test)

print('\n Model Evalutal of XG Boost')
print('\n Confusion matrix XG BOOST :-', confusion_matrix(y_test,y_pred_xg))
print('Classification Report of XG boost :-', classification_report(y_test,y_pred_xg))
print('ROC_AUC Score Xg BOOST :-', roc_auc_score(y_test,pipeline_xg.predict_proba(x_test)[:,1]))


#Model 5 : XG Boost Adasyn Untuned

pipline_xg_adasyn = ImbPipeline(steps=[
    ('preprocessor', preprocessor),
    ('adasyn', ADASYN(random_state=42)),
    ('classifier', XGBClassifier(n_estimators=5,use_label_encoder=False,eval_metrics='logloss'))
])

pipline_xg_adasyn.fit(x_train,y_train)
y_pred_xg_adasyn = pipline_xg_adasyn.predict(x_test)

print('Model Evaluation of XG Boost Adasyn')
print('\n Confusion Matrix of XG Boost Adasyn :-', confusion_matrix(y_test,y_pred_xg_adasyn))
print('Classification Report of XG boost adasyn :-', classification_report(y_test,y_pred_xg_adasyn))
print('ROC_AUC SCORE of XG boost Adasyn :-', roc_auc_score(y_test,pipline_xg_adasyn.predict_proba(x_test)[:,1]))


#Hyperparamter Tuning for XGBoost with ADASYN

pipeline_xg_abasyn_tune = ImbPipeline(steps=[
    ('preprocessor',preprocessor),
    ('adasyn', ADASYN(random_state=42)),
    ('classifier', XGBClassifier(random_state=42,use_label_encoder=False,eval_metric='logloss'))
])

param_grid_xg_adasyn = {
    'adasyn__n_neighbors' : [5],
    'classifier__n_estimators' : [100],
    'classifier__learning_rate' : [0.05,0.1],
    'classifier__max_depth' : [5,7],
    'classifier__subsample' : [0.8,1.0]
} 

grid_search_xg_adasyn = GridSearchCV(
    estimator=pipeline_xg_abasyn_tune,
    param_grid=param_grid_xg_adasyn,
    cv=cv_splitter,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search_xg_adasyn.fit(x_train,y_train)

print('\n Best Parameters for XG Boost Adasyn Tuned :-', grid_search_xg_adasyn.best_params_)
print('\n Best cross validation ROC AUC for XG BOOST TUNED :-', grid_search_xg_adasyn.best_score_)

best_xg_adasyn_model = grid_search_xg_adasyn.best_estimator_

y_pred_best_xg = best_xg_adasyn_model.predict(x_test)

print('\n\n ----- Model Evaluation of XG Boost with ADASYN Tuned ----')
print('Classification Report :-', classification_report(y_test,y_pred_best_xg))
print('Confusion Matrix XG Boost Tuned :-', confusion_matrix(y_test,y_pred_best_xg))
print('ROC AUC Score :-', roc_auc_score(y_test,best_xg_adasyn_model.predict_proba(x_test)[:,1]))

xgb_classifier = best_xg_adasyn_model.named_steps['classifier']
features_importances_xgb_adasyn = xgb_classifier.feature_importances_

feature_names = x.columns.to_list()

importance_df_xgb = pd.DataFrame({
    'Feature' : feature_names,
    'Importance' : features_importances_xgb_adasyn
}).sort_values(by='Importance',ascending=False)



print('\n Top 10 Features Important (XG BOOST ADASYN)')
print(importance_df_xgb.head(10))

#Plotting
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_xgb.head(20), palette='magma') # Show top 20
plt.title('Top 20 Feature Importances for XGBoost (ADASYN)', fontsize=16)
plt.xlabel('Importance (Gain)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

#Model 6 : Random Forest Classifier
pipeline_rfc = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier',RandomForestClassifier(random_state=42,n_estimators=100))
])
pipeline_rfc.fit(x_train,y_train)
y_pred_pipeline_rfc = pipeline_rfc.predict(x_test)

print('\n\n Model Evaluation of Random Forest classifier')
print('Confusion Matrix of RFC :-', confusion_matrix(y_test,y_pred_pipeline_rfc))
print('Classification Report of RFC :- ', classification_report(y_test,y_pred_pipeline_rfc))
print('ROC_AUC Score of RFC :-' , roc_auc_score(y_test,pipeline_rfc.predict_proba(x_test)[:,1]))

#Model 7 : RFC with SMOTE  - Original Untuned

pipeline_rfc_smote = ImbPipeline(steps=[
    ('preprocessor',preprocessor),
    ('smote',SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42,n_estimators=100))
])
pipeline_rfc_smote.fit(x_train,y_train)
y_pred_rfc_smote =  pipeline_rfc_smote.predict(x_test)

print('\n Model Evalutaion of RFC SMOTE')
print('Confusion Matrix of RFC SMOTE :-', confusion_matrix(y_test,y_pred_rfc_smote))
print('Classification Report of RFC :-', classification_report(y_test,y_pred_rfc_smote))
print('ROC_AUC Score of RFC SMOTE :-', roc_auc_score(y_test,pipeline_rfc_smote.predict_proba(x_test)[:,1]))


# Hyper Parameter tuning for RFC with Smote

pipeline_rfc_smote_tune = ImbPipeline(steps=[
    ('preprocessor',preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier',RandomForestClassifier(random_state=42))
])

param_grid_rfc_smote = {
    'smote__k_neighbors' : [5],
    'classifier__n_estimators' : [100],
    'classifier__max_depth' : [None],
    'classifier__min_samples_split' : [2],
    'classifier__min_samples_leaf' : [1]
}

grid_search_rfc_smote = GridSearchCV(
    estimator=pipeline_rfc_smote_tune,
    param_grid=param_grid_rfc_smote,
    cv=cv_splitter,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search_rfc_smote.fit(x_train,y_train)

print('\n Best parameters for RFC Smote :-', grid_search_rfc_smote.best_params_)
print('Best cross-validated for ROC AUC FOR RFC Smote',grid_search_rfc_smote.best_score_)

best_rfc_smote_model = grid_search_rfc_smote.best_estimator_

y_pred_best_rfc_smote = best_rfc_smote_model.predict(x_test)

print('Model Evaluation of RFC SMOTE with hyperparameter tuning')
print('\n Classfication report :-', classification_report(y_test,y_pred_best_rfc_smote))
print('\n Confusion Matrix :-', confusion_matrix(y_test,y_pred_best_rfc_smote))
print('\n ROC AUC Score :-', roc_auc_score(y_test,best_rfc_smote_model.predict_proba(x_test)[:,1]))


rfc_classifier = best_rfc_smote_model.named_steps['classifier']
feature_importances_rfc = rfc_classifier.feature_importances_


importance_df_rfc = pd.DataFrame({
    'Feature' : feature_names,
    'Importance' : feature_importances_rfc
}).sort_values(by='Importance',ascending=False)

# TOP 10 important features
print('\n Top 10 Important features (Random Forest)')
print(importance_df_rfc.head(10))

#Plotting
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df_rfc.head(20), palette='viridis') # Show top 20
plt.title('Top 20 Feature Importances for Random Forest (SMOTE)', fontsize=16)
plt.xlabel('Importance (Mean Decrease in Impurity)', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


#ROC Curves 

plt.figure(figsize=(10,8))

y_proba_lr = best_lr_model.predict_proba(x_test)[:,1]
y_proba_xg_adasyn = best_xg_adasyn_model.predict_proba(x_test)[:,1]
y_proba_rfc_smote = best_rfc_smote_model.predict_proba(x_test)[:,1]


#Plotting ROC For Logistic Regressor
fpr_lr,tpr_lr,_  = roc_curve(y_test,y_proba_lr)
roc_auc_lr = auc(fpr_lr,tpr_lr)
plt.plot(fpr_lr,tpr_lr,color='blue',lw=2,label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')

# PLottin ROC for XG BOOST with Adasyn Tuned
fpr_xg_adasyn, tpr_xg_adasyn, _ = roc_curve(y_test, y_proba_xg_adasyn)
roc_auc_xg_adasyn = auc(fpr_xg_adasyn, tpr_xg_adasyn)
plt.plot(fpr_xg_adasyn, tpr_xg_adasyn, color='red', lw=2, label=f'XGBoost (ADASYN) (AUC = {roc_auc_xg_adasyn:.2f})')


# Plot ROC for Tuned RFC with SMOTE
fpr_rfc_smote, tpr_rfc_smote, _ = roc_curve(y_test, y_proba_rfc_smote)
roc_auc_rfc_smote = auc(fpr_rfc_smote, tpr_rfc_smote)
plt.plot(fpr_rfc_smote, tpr_rfc_smote, color='green', lw=2, label=f'Random Forest (SMOTE) (AUC = {roc_auc_rfc_smote:.2f})')


plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curves for Fraud Detection Models', fontsize=16)
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

plt.show()


#Saving The Best Model as JobLib
import joblib

joblib.dump(best_xg_adasyn_model,'best_fraud_detection_model.joblib')

joblib.dump(x.columns.tolist(),'feature_columns.joblib')

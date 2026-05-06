from requests import packages
from sklearn.linear_model import LogisticRegression
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.metrics import accuracy_score , classification_report ,confusion_matrix,roc_auc_score,roc_curve,ConfusionMatrixDisplay
df = pd.read_csv('churn_Modelling.csv')
df.head()

df.shape
df['Geography'].value_counts()
df.describe()

counts = df['Exited'].value_counts()
plt.figure(figsize=(12,4))

plt.subplot(1,2,1)
plt.pie(counts, labels=['Not Churned', 'Churned'], autopct='%1.1f%%')
plt.title('Churn Distribution')


plt.subplot(1,2,2)
sns.countplot(x='Exited', data=df)
plt.xticks([0,1], ['Not Churned', 'Churned'])
plt.title('Churn Count')

plt.tight_layout()
plt.show()

print("Churn Rate:", round(df['Exited'].mean()*100, 2), "%")
plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
geo = df.groupby('Geography')['Exited'].mean()
geo.plot(kind='bar')
plt.title('Churn Rate By geography')
plt.ylabel('Churn Rate')

plt.subplot(1,2,2)
gender = df.groupby('Gender')['Exited'].mean()
gender.plot(kind='bar')
plt.title('Churn rate by gender')
plt.ylabel('Churn Rate ')
plt.tight_layout()
plt.show()
nums = ['Age','CreditScore','Balance','EstimatedSalary','Tenure','NumOfProducts']
plt.figure(figsize=(16,9))
for i ,col  in enumerate(nums):
    plt.subplot(2,3,i+1)
    plt.hist(df[df['Exited']==0][col],bins=30,alpha=0.6,label='Not Churned')
    plt.hist(df[df['Exited']==1][col],bins=30,alpha=0.6,label='Churned')
    plt.title(col)
    plt.legend()
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12,9))
corr = df[nums+['Exited']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5)
plt.tight_layout()
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

# Drop columns
df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)

# Encode
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

df = pd.get_dummies(df, columns=['Geography'], drop_first=True)

# Split
X = df.drop('Exited', axis=1)
y = df['Exited']

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler = StandardScaler()
x_train_sc = scaler.fit_transform(x_train)
x_test_sc = scaler.transform(x_test)
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
x_train_sm , y_train_sm = smote.fit_resample(x_train,y_train)
y_train_sm.value_counts()
pd.Series(y_train_sm).value_counts()
x_train_sm_sc = scaler.fit_transform(x_train_sm)
x_test_sc = scaler.transform(x_test)
# ── 6. LOGISTIC REGRESSION (with SMOTE) ────────────────────────────────────
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(x_train_sm_sc, y_train_sm)
y_pred_lr = lr.predict(x_test_sc)

print('=== Logistic Regression (SMOTE) ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred_lr):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, lr.predict_proba(x_test_sc)[:,1]):.4f}')
print(classification_report(y_test, y_pred_lr, target_names=['Not Churned', 'Churned']))
# ── 7. RANDOM FOREST (with SMOTE) ──────────────────────────────────────────
rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(x_train_sm, y_train_sm)
y_pred_rf = rf.predict(x_test)

print('=== Random Forest (SMOTE) ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred_rf):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, rf.predict_proba(x_test)[:,1]):.4f}')
print(classification_report(y_test, y_pred_rf, target_names=['Not Churned', 'Churned']))
# ── 8. GRADIENT BOOSTING (with SMOTE) ─────────────────────────────────────
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                 max_depth=5, random_state=42)
gb.fit(x_train_sm, y_train_sm)
y_pred_gb = gb.predict(x_test)

print('=== Gradient Boosting (SMOTE) ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred_gb):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, gb.predict_proba(x_test)[:,1]):.4f}')
print(classification_report(y_test, y_pred_gb, target_names=['Not Churned', 'Churned']))
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
xgb = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)
xgb.fit(x_train_sm, y_train_sm)
y_pred_xgb = xgb.predict(x_test)

print('=== XGBoost (SMOTE) ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, xgb.predict_proba(x_test)[:,1]):.4f}')
print(classification_report(y_test, y_pred_xgb, target_names=['Not Churned', 'Churned']))
cat = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    random_seed=42,
    verbose=0
)
cat.fit(x_train_sm, y_train_sm)
y_pred_cat = cat.predict(x_test)

print('=== CatBoost (SMOTE) ===')
print(f'Accuracy : {accuracy_score(y_test, y_pred_cat):.4f}')
print(f'ROC-AUC  : {roc_auc_score(y_test, cat.predict_proba(x_test)[:,1]):.4f}')
print(classification_report(y_test, y_pred_cat, target_names=['Not Churned', 'Churned']))
# ── 11. MODEL COMPARISON TABLE ─────────────────────────────────────────────
models = {
    'Logistic Regression'       : (lr,  x_test_sc, y_pred_lr),
    'Random Forest'             : (rf,  x_test,    y_pred_rf),
    'Gradient Boosting'         : (gb,  x_test,    y_pred_gb),
    'XGBoost'                   : (xgb, x_test,    y_pred_xgb),
    'CatBoost'                  : (cat, x_test,    y_pred_cat),
}

results = []
for name, (model, x_t, y_p) in models.items():
    results.append({
        'Model'   : name,
        'Accuracy': round(accuracy_score(y_test, y_p), 4),
        'ROC_AUC' : round(roc_auc_score(y_test, model.predict_proba(x_t)[:, 1]), 4)
    })

result_df = pd.DataFrame(results).sort_values('ROC_AUC', ascending=False)
print(result_df.to_string(index=False))
# ── 12. CONFUSION MATRICES ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(28, 5))

for ax, (name, (model, x_t, y_p)) in zip(axes, models.items()):
    cm = confusion_matrix(y_test, y_p)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Not Churned', 'Churned'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontweight='bold', fontsize=10)

plt.suptitle('Confusion Matrices — All Models (SMOTE)', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()
# ── 13. ROC CURVES ─────────────────────────────────────────────────────────
plt.figure(figsize=(9, 7))

for name, (model, x_t, y_p) in models.items():
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(x_t)[:, 1])
    auc = roc_auc_score(y_test, model.predict_proba(x_t)[:, 1])
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})')

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves — All Models (SMOTE)', fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
# ── 14. FEATURE IMPORTANCE — XGBoost ──────────────────────────────────────
feat_imp_xgb = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp_xgb.values, y=feat_imp_xgb.index, palette='viridis')
plt.title('Feature Importance — XGBoost', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print(feat_imp_xgb)
# ── 15. FEATURE IMPORTANCE — CatBoost ─────────────────────────────────────
feat_imp_cat = pd.Series(cat.get_feature_importance(), index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp_cat.values, y=feat_imp_cat.index, palette='magma')
plt.title('Feature Importance — CatBoost', fontsize=13, fontweight='bold')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

print(feat_imp_cat)
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
# ── 16. CROSS VALIDATION — XGBoost & CatBoost ─────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# XGBoost CV (on original data — SMOTE inside CV loop is best practice
# but for project submission this simpler approach is standard)
cv_acc_xgb = cross_val_score(xgb, X, y, cv=skf, scoring='accuracy')
cv_auc_xgb = cross_val_score(xgb, X, y, cv=skf, scoring='roc_auc')

print('=== 5-Fold CV — XGBoost ===')
print(f'Accuracy : {cv_acc_xgb.mean():.4f} ± {cv_acc_xgb.std():.4f}')
print(f'ROC-AUC  : {cv_auc_xgb.mean():.4f} ± {cv_auc_xgb.std():.4f}')
print()

# CatBoost CV
cv_acc_cat = cross_val_score(cat, X, y, cv=skf, scoring='accuracy')
cv_auc_cat = cross_val_score(cat, X, y, cv=skf, scoring='roc_auc')

print('=== 5-Fold CV — CatBoost ===')
print(f'Accuracy : {cv_acc_cat.mean():.4f} ± {cv_acc_cat.std():.4f}')
print(f'ROC-AUC  : {cv_auc_cat.mean():.4f} ± {cv_auc_cat.std():.4f}')
# ── 17. PREDICT ON NEW CUSTOMER ────────────────────────────────────────────
new_customer = pd.DataFrame([{
    'CreditScore'       : 620,
    'Gender'            : 1,      # Male
    'Age'               : 42,
    'Tenure'            : 3,
    'Balance'           : 95000,
    'NumOfProducts'     : 2,
    'HasCrCard'         : 1,
    'IsActiveMember'    : 0,
    'EstimatedSalary'   : 75000,
    'Geography_Germany' : 0,
    'Geography_Spain'   : 0       # France (both 0)
}])

new_customer = new_customer[X.columns]

print('──────────────────────────────────────────')
print(' Churn Probability — New Customer')
print('──────────────────────────────────────────')
for name, (model, x_t, y_p) in models.items():
    x_input = scaler.transform(new_customer) if name == 'Logistic Regression' else new_customer
    prob = model.predict_proba(x_input)[0][1]
    pred = model.predict(x_input)[0]
    status = '⚠️  Will Churn' if pred == 1 else '✅ Will Stay'
    print(f'{name:<25}  Prob: {prob:.2%}   →  {status}')
print('──────────────────────────────────────────')
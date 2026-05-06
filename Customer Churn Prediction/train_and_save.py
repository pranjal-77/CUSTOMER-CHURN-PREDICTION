import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

def main():
    print("Loading data...")
    df = pd.read_csv('Churn_Modelling.csv')
    
    print("Preprocessing data...")
    # Drop columns
    df = df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1)
    
    # Encode Gender
    le = LabelEncoder()
    df['Gender'] = le.fit_transform(df['Gender'])
    
    # One-hot encode Geography
    df = pd.get_dummies(df, columns=['Geography'], drop_first=True)
    
    # Split features and target
    X = df.drop('Exited', axis=1)
    y = df['Exited']
    
    print(f"Features: {list(X.columns)}")
    
    # Train test split
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    x_train_sm, y_train_sm = smote.fit_resample(x_train, y_train)
    
    print("Training XGBoost model...")
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
    
    # Save the model
    print("Saving model to xgboost_model.pkl...")
    joblib.dump(xgb, 'xgboost_model.pkl')
    
    print("Model saved successfully!")

if __name__ == '__main__':
    main()

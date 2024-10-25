from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    # Hyperparameter tuning using GridSearchCV for Logistic Regression
    lr_param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10],
        'max_iter': [100, 200, 300]
    }

    lr_model = LogisticRegression(solver='liblinear')
    lr_grid_search = GridSearchCV(lr_model, lr_param_grid, scoring='f1', cv=StratifiedKFold(n_splits=5), verbose=1)
    lr_grid_search.fit(X_train, y_train)
    lr_best_model = lr_grid_search.best_estimator_

    # Hyperparameter tuning for Random Forest
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }

    rf_model = RandomForestClassifier()
    rf_grid_search = GridSearchCV(rf_model, rf_param_grid, scoring='f1', cv=StratifiedKFold(n_splits=5), verbose=1)
    rf_grid_search.fit(X_train, y_train)
    rf_best_model = rf_grid_search.best_estimator_

    # Hyperparameter tuning for XGBoost
    xgb_param_grid = {
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
    }

    xgb_model = XGBClassifier(eval_metric='logloss', use_label_encoder=False)
    xgb_grid_search = GridSearchCV(xgb_model, xgb_param_grid, scoring='f1', cv=StratifiedKFold(n_splits=5), verbose=1)
    xgb_grid_search.fit(X_train, y_train)
    xgb_best_model = xgb_grid_search.best_estimator_

    # Predictions for each model
    lr_pred = lr_best_model.predict(X_test)
    rf_pred = rf_best_model.predict(X_test)
    xgb_pred = xgb_best_model.predict(X_test)

    # Evaluation metrics
    lr_report = classification_report(y_test, lr_pred)
    rf_report = classification_report(y_test, rf_pred)
    xgb_report = classification_report(y_test, xgb_pred)

    lr_auc = roc_auc_score(y_test, lr_best_model.predict_proba(X_test)[:, 1])
    rf_auc = roc_auc_score(y_test, rf_best_model.predict_proba(X_test)[:, 1])
    xgb_auc = roc_auc_score(y_test, xgb_best_model.predict_proba(X_test)[:, 1])

    print("Logistic Regression Classification Report:\n", lr_report)
    print("Logistic Regression AUC Score:", lr_auc)

    print("Random Forest Classification Report:\n", rf_report)
    print("Random Forest AUC Score:", rf_auc)

    print("XGBoost Classification Report:\n", xgb_report)
    print("XGBoost AUC Score:", xgb_auc)

    # Return the best models from grid search for further use
    return lr_best_model, rf_best_model, xgb_best_model
